"""Stock tracker update script (version 2).

Runs from GitHub Actions. For every company in config/companies.json it:

  every run    market status, quotes for the company, benchmarks and peers,
               news from Finnhub, Yahoo Finance and Google News (plus Alpha
               Vantage every few hours), a free keyword filter, AI sorting of
               the new headlines that pass it, and the 30 second brief only
               when the news or the price has really changed
  once a day   profile, key metrics, analyst ratings, grades and price targets,
               estimates, earnings history and calendar, customer earnings dates,
               insider trades, valuation ratios, quarterly results, a year of
               daily prices for the company and its benchmark funds, and the AI
               outlook when something has changed since the last one
  big news     the outlook is rewritten early when a new high importance story lands

AI use is capped by a daily token budget set in config/companies.json. Optional
jobs are skipped first as the budget fills up, and every run's usage is logged in
data/_usage.json.

Everything is saved to data/<TICKER>.json for the app page.
Which endpoints work on the free plans is remembered in data/_capabilities.json.
"""

import json
import math
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ai import AI, AIError
from sources import Sources, days_ago_iso, headline_key, merge_news, number, today_iso, utc_now

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "companies.json"
DATA_DIR = ROOT / "data"
CAPABILITIES_PATH = DATA_DIR / "_capabilities.json"
USAGE_PATH = DATA_DIR / "_usage.json"

NEWS_DAYS = 7
MAJOR_EVENT_DAYS = 365
MAX_NEW_ITEMS_PER_RUN = 60
MAX_SORT_ATTEMPTS = 3
BRIEF_MAX_AGE_HOURS = 6
USAGE_DAYS_KEPT = 60
IMPORTANCE_WEIGHT = {"high": 3, "medium": 2, "low": 1}

# How full the daily budget may be before each AI job is skipped (percent).
# News sorting matters most, so it runs longest. Extra outlooks go first.
BUDGET_LIMITS = {"news": 100, "brief": 90, "outlook": 90, "outlook_extra": 60}

# Headlines that look like generic stock lists are dropped without asking the AI.
LISTICLE = re.compile(
    r"\b(stocks? to (buy|watch|own|hold)|best (ai |tech |growth |chip )?stocks|top \d+ (ai |tech )?stocks|"
    r"should you buy|millionaire|\d+ (ai |tech |growth |chip )?stocks)\b",
    re.IGNORECASE,
)
ALPHA_NEWS_HOURS_PER_COMPANY = 2
EARNINGS_HOURS = {"amc": "after the US close", "bmo": "before the US open", "dmh": "during US trading hours"}


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def load_json(path, fallback):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return fallback
    return fallback


def save_json(path, data):
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def get_key(name):
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"Missing environment variable {name}. Check the GitHub secret name.")
    return value


def hours_since(iso_text):
    try:
        then = datetime.fromisoformat(iso_text)
    except (TypeError, ValueError):
        return 1e9
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (utc_now() - then).total_seconds() / 3600


# ---------------------------------------------------------------------------
# Number crunching
# ---------------------------------------------------------------------------

def forward_eps(years, today):
    """Next twelve months EPS: blend this fiscal year and next by time remaining."""
    if len(years) < 2 or years[0].get("eps_avg") is None or years[1].get("eps_avg") is None:
        return None
    year_end = date.fromisoformat(years[0]["fiscal_year_end"][:10])
    days_left = (year_end - date.fromisoformat(today)).days
    weight = max(0.0, min(1.0, days_left / 365))
    return round(weight * years[0]["eps_avg"] + (1 - weight) * years[1]["eps_avg"], 2)


def update_eps_history(history, today, years):
    history = [h for h in history if h.get("date") != today]
    history.append({"date": today, "eps": {y["fiscal_year_end"]: y.get("eps_avg") for y in years}})
    return history[-400:]


def revision_pct(history, fiscal_year_end, today, days=30):
    cutoff = (date.fromisoformat(today) - timedelta(days=days)).isoformat()
    points = [h["eps"][fiscal_year_end] for h in history
              if h.get("date", "") >= cutoff and (h.get("eps") or {}).get(fiscal_year_end)]
    if len(points) < 2:
        return None
    return round((points[-1] - points[0]) / points[0] * 100, 2)


def insider_summary(transactions, days=90):
    cutoff = days_ago_iso(days)
    sold = bought = 0.0
    sellers, buyers = set(), set()
    for t in transactions or []:
        if (t.get("transactionDate") or "") < cutoff:
            continue
        value = abs(number(t.get("change")) or 0) * (number(t.get("transactionPrice")) or 0)
        if t.get("transactionCode") == "S":
            sold += value
            sellers.add(t.get("name"))
        elif t.get("transactionCode") == "P":
            bought += value
            buyers.add(t.get("name"))
    return {"days": days, "sold_usd": round(sold), "bought_usd": round(bought),
            "sellers": len(sellers), "buyers": len(buyers)}


def merge_series(old, new, limit=400):
    """Combine two [[date, close], ...] lists, newer values win."""
    by_date = {d: c for d, c in old or [] if c}
    by_date.update({d: c for d, c in new or [] if c})
    return [[d, by_date[d]] for d in sorted(by_date)][-limit:]


def adjust_splits(series, symbol, splits):
    """Puts prices from before a known split on the new share scale.

    splits comes from the settings file, for example
    {"symbol": "SOXX", "date": "2026-11-05", "ratio": 3} for a 3 for 1 split.
    Prices dated before the split are divided by the ratio, but only while the
    series still shows the jump (checked on the last price before the split and
    the first price after it), so running this again never divides twice."""
    series = [list(p) for p in series or [] if isinstance(p, (list, tuple)) and len(p) == 2 and p[1]]
    for split in splits or []:
        if not isinstance(split, dict) or split.get("symbol") != symbol:
            continue
        when, ratio = split.get("date"), split.get("ratio")
        if not isinstance(when, str) or not isinstance(ratio, (int, float)) or ratio <= 0 or ratio == 1:
            continue
        before = [p for p in series if p[0] < when]
        after = [p for p in series if p[0] >= when]
        if not before or not after:
            continue
        jump = before[-1][1] / after[0][1]
        # Adjust only when the jump is closer to the split ratio than to no change.
        if jump <= 0 or abs(math.log(jump / ratio)) >= abs(math.log(jump)):
            continue
        series = [[d, round(c / ratio, 4)] for d, c in before] + after
    return series


def recurring_dates(events, days_ahead=75):
    """Turns {"label": ..., "day_of_month": 10} into the next real dates."""
    out = []
    today = date.fromisoformat(today_iso())
    for event in events or []:
        day = event.get("day_of_month")
        if not isinstance(day, int) or not 1 <= day <= 28:
            continue
        month_start = today.replace(day=1)
        for step in range(3):
            year = month_start.year + (month_start.month - 1 + step) // 12
            month = (month_start.month - 1 + step) % 12 + 1
            when = date(year, month, day)
            if today <= when <= today + timedelta(days=days_ahead):
                out.append({"date": when.isoformat(), "label": event.get("label", "Event"),
                            "kind": event.get("kind", "industry"), "approx": True,
                            "why": event.get("why", "")})
    return out


# ---------------------------------------------------------------------------
# News handling
# ---------------------------------------------------------------------------

def prune_items(items):
    cutoff = days_ago_iso(NEWS_DAYS)
    return {k: v for k, v in items.items() if (v.get("published") or v.get("first_seen") or "")[:10] >= cutoff}


def news_keywords(company):
    """Returns (names, industry words): names always pass the filter, industry words pass unless it is a stock list."""
    names = [w.lower() for w in (company.get("name"), company.get("ticker")) if w]
    industry = [w.lower() for w in company.get("news_keywords") or [] if w]
    return names, industry


def worth_sorting(item, keywords):
    """Free check before the AI. Any headline naming the company is kept. Others are kept only
    when they mention the industry and are not a generic list of stocks."""
    names, industry = keywords
    headline = (item.get("headline") or "").lower()
    if any(n in headline for n in names):
        return True
    if LISTICLE.search(headline):
        return False
    words = headline + " " + (item.get("summary") or "").lower()
    return any(k in words for k in names + industry)


def price_move_pct(old_price, new_price):
    old_price, new_price = number(old_price), number(new_price)
    if not old_price or not new_price:
        return None
    return abs(new_price / old_price - 1) * 100


def important_news_since(items, since_iso):
    """True when a medium or high importance relevant headline was first seen after since_iso."""
    since_iso = since_iso or ""
    return any(i.get("relevant") and i.get("importance") in ("high", "medium")
               and (i.get("first_seen") or "") > since_iso for i in items.values())


def build_stories(items, titles):
    groups = {}
    for item in items.values():
        if not item.get("sorted") or not item.get("relevant"):
            continue
        groups.setdefault(item["story"], []).append(item)

    stories = []
    for key, members in groups.items():
        members.sort(key=lambda i: (IMPORTANCE_WEIGHT.get(i["importance"], 1), i.get("published") or ""), reverse=True)
        lead = members[0]
        impacts = [m["impact"] for m in members]
        sources, seen = [], set()
        for m in sorted(members, key=lambda i: i.get("published") or "", reverse=True):
            if m["source"] in seen:
                continue
            seen.add(m["source"])
            sources.append({"source": m["source"], "url": m["url"], "headline": m["headline"],
                            "published": m.get("published")})
        latest = max((m.get("published") or m.get("first_seen") or "") for m in members)
        first_seen = min((m.get("first_seen") or latest) for m in members)
        hours = hours_since(latest)
        score = IMPORTANCE_WEIGHT.get(lead["importance"], 1) * 10 + len(members) * 2 - min(hours, 168) / 12
        stories.append({
            "key": key,
            "title": titles.get(key) or lead["headline"],
            "summary": lead.get("ai_summary") or "",
            "impact": max(set(impacts), key=impacts.count) if lead["impact"] == "neutral" else lead["impact"],
            "importance": lead["importance"],
            "topic": lead["topic"],
            "latest": latest,
            "first_seen": first_seen,
            "count": len(members),
            "image": next((m.get("image") for m in members if m.get("image")), None),
            "sources": sources[:8],
            "score": round(score, 2),
        })
    stories.sort(key=lambda s: s["score"], reverse=True)
    return stories


def story_digest(stories, limit):
    return [{"key": s["key"], "title": s["title"], "summary": s["summary"], "impact": s["impact"],
             "importance": s["importance"], "topic": s["topic"], "latest": s["latest"], "articles": s["count"]}
            for s in stories[:limit]]


def update_major_events(previous, stories):
    events = {e["key"]: e for e in previous or []}
    for s in stories:
        if s["importance"] == "high":
            events[s["key"]] = {"key": s["key"], "date": (s["latest"] or today_iso())[:10], "title": s["title"],
                                "impact": s["impact"], "url": s["sources"][0]["url"] if s["sources"] else None}
    cutoff = days_ago_iso(MAJOR_EVENT_DAYS)
    return sorted((e for e in events.values() if e["date"] >= cutoff), key=lambda e: e["date"])


# ---------------------------------------------------------------------------
# One company
# ---------------------------------------------------------------------------

def update_company(company, config, src, ai, market):
    ticker = company["ticker"]
    path = DATA_DIR / f"{ticker}.json"
    prev = load_json(path, {})
    today = today_iso()
    now_iso = utc_now().isoformat()
    errors = []
    daily_due = prev.get("daily_done") != today or prev.get("version") != 2

    # ----- prices every run -------------------------------------------------
    quote = src.quote(ticker)  # if this fails the whole company is skipped and the old file stays
    print(f"{ticker} {quote['price']} ({quote['change_pct']:+.2f}%)")

    benchmarks = {}
    for symbol in company.get("benchmarks", []):
        q = src.safe("finnhub.quote", src.quote, symbol)
        if q:
            benchmarks[symbol] = {"price": q["price"], "change_pct": q["change_pct"],
                                  "relative_pct": round(quote["change_pct"] - q["change_pct"], 2)}
    peers = {}
    for symbol in company.get("peers", []):
        q = src.safe("finnhub.quote", src.quote, symbol)
        if q:
            peers[symbol] = {"price": q["price"], "change_pct": q["change_pct"]}

    # ----- daily data ---------------------------------------------------------
    keep = lambda name, default: prev.get(name, default)
    profile = keep("profile", {})
    fundamentals = keep("fundamentals", {})
    analysts = keep("analysts", {})
    earnings = keep("earnings", {})
    insiders = keep("insiders", {})
    estimates = keep("estimates", {})
    eps_history = keep("eps_history", [])
    history = keep("history", {})
    press = keep("press_releases", [])
    catalysts = keep("catalysts", [])

    if daily_due:
        print(f"{ticker} daily refresh")
        profile = src.safe("finnhub.profile", src.profile, ticker, default=profile) or profile
        basic = src.safe("finnhub.basic_financials", src.basic_financials, ticker)
        fundamentals = {
            "metrics": (basic or {}).get("metrics") or fundamentals.get("metrics", {}),
            "quarterly": (basic or {}).get("quarterly") or fundamentals.get("quarterly", {}),
            "ratios_ttm": src.safe("fmp.ratios_ttm", src.ratios_ttm, ticker) or fundamentals.get("ratios_ttm", {}),
            "key_metrics_ttm": src.safe("fmp.key_metrics_ttm", src.key_metrics_ttm, ticker) or fundamentals.get("key_metrics_ttm", {}),
            "quarterly_income": src.safe("fmp.quarterly_income", src.quarterly_income, ticker) or fundamentals.get("quarterly_income", []),
        }
        analysts = {
            "trends": src.safe("finnhub.recommendation_trends", src.recommendation_trends, ticker) or analysts.get("trends", []),
            "consensus": src.safe("fmp.grades_consensus", src.grades_consensus, ticker) or analysts.get("consensus"),
            "latest_grades": src.safe("fmp.latest_grades", src.latest_grades, ticker) or analysts.get("latest_grades", []),
            "upgrades": src.safe("finnhub.upgrades_downgrades", src.upgrades_downgrades, ticker) or analysts.get("upgrades", []),
            "price_target": (src.safe("fmp.price_target_consensus", src.price_target_consensus, ticker)
                             or src.safe("finnhub.price_target", src.finnhub_price_target, ticker)
                             or analysts.get("price_target")),
            "price_target_summary": src.safe("fmp.price_target_summary", src.price_target_summary, ticker) or analysts.get("price_target_summary"),
        }
        calendar = src.safe("finnhub.earnings_calendar", src.earnings_calendar, ticker)
        if calendar is None:
            calendar = earnings.get("calendar", [])
        earnings = {
            "history": src.safe("finnhub.earnings_history", src.earnings_history, ticker) or earnings.get("history", []),
            "calendar": calendar,
            "next": next((c for c in calendar if (c.get("date") or "") >= today), None),
        }
        transactions = src.safe("finnhub.insider_transactions", src.insider_transactions, ticker)
        if transactions is None:
            transactions = insiders.get("transactions", [])
        insiders = {
            "transactions": transactions,
            "summary": insider_summary(transactions),
            "sentiment": src.safe("finnhub.insider_sentiment", src.insider_sentiment, ticker) or insiders.get("sentiment", []),
        }
        years = src.safe("fmp.analyst_estimates", src.analyst_estimates, ticker)
        if years:
            estimates = {"fetched_on": today, "years": years}
            eps_history = update_eps_history(eps_history, today, years)

        company_prices = src.safe("fmp.price_history", src.price_history, ticker)
        history = {"company": company_prices or history.get("company", []),
                   "benchmarks": dict(history.get("benchmarks", {}))}

        press = src.safe("fmp.press_releases", src.press_releases, ticker) or press

        catalysts = []
        if earnings.get("next"):
            n = earnings["next"]
            when = EARNINGS_HOURS.get(n.get("hour"), "time not confirmed")
            catalysts.append({"date": n["date"], "label": f"{ticker} earnings", "kind": "earnings",
                              "approx": False, "why": f"Quarterly results and guidance, {when}"})
        for customer in company.get("customers", []):
            rows = src.safe("finnhub.earnings_calendar", src.earnings_calendar, customer) or []
            upcoming = next((r for r in rows if (r.get("date") or "") >= today), None)
            if upcoming:
                catalysts.append({"date": upcoming["date"], "label": f"{customer} earnings", "kind": "customer",
                                  "approx": False, "why": "Watch its spending plans for AI chips"})
        catalysts += recurring_dates(company.get("recurring_events"))
        catalysts.sort(key=lambda c: c["date"])

    # Fund price history: Alpha Vantage at most once a day per fund (it allows 25 calls a day),
    # and each run adds the live quote so the series keeps growing even without it.
    quote_day = (quote.get("time") or now_iso)[:10]
    bench_series = dict(history.get("benchmarks", {}))
    splits = config.get("splits") or []
    for symbol, b in benchmarks.items():
        points = src.safe("alphavantage.daily", src.alpha_daily, symbol, min_hours=20, cache_key=symbol)
        # Alpha Vantage free prices are not split adjusted, so fix the saved history and the new
        # download separately before merging them.
        saved = adjust_splits(bench_series.get(symbol), symbol, splits)
        fresh = adjust_splits((points or []) + [[quote_day, b["price"]]], symbol, splits)
        bench_series[symbol] = merge_series(saved, fresh)
    history = {**history, "benchmarks": bench_series}

    # ----- valuation ----------------------------------------------------------
    years = estimates.get("years", [])
    eps_ntm = forward_eps(years, today)
    next_year = years[1] if len(years) > 1 else None
    metrics = fundamentals.get("metrics", {})
    ratios = fundamentals.get("ratios_ttm", {})
    key_metrics = fundamentals.get("key_metrics_ttm", {})
    target = (analysts.get("price_target") or {})
    target_mid = number(target.get("targetConsensus") or target.get("targetMean"))
    valuation = {
        "forward_eps": eps_ntm,
        "forward_pe": round(quote["price"] / eps_ntm, 1) if eps_ntm and eps_ntm > 0 else None,
        "next_fiscal_year_end": next_year["fiscal_year_end"] if next_year else None,
        "next_year_eps": next_year.get("eps_avg") if next_year else None,
        "next_year_revision_30d_pct": revision_pct(eps_history, next_year["fiscal_year_end"], today) if next_year else None,
        "pe_ttm": number(metrics.get("peTTM") or metrics.get("peBasicExclExtraTTM") or ratios.get("priceToEarningsRatioTTM")),
        "peg": number(ratios.get("priceToEarningsGrowthRatioTTM") or metrics.get("pegTTM")),
        "fcf_yield_pct": round(key_metrics["freeCashFlowYieldTTM"] * 100, 2) if key_metrics.get("freeCashFlowYieldTTM") else None,
        "ev_ebitda": number(key_metrics.get("evToEBITDATTM")),
        "target_upside_pct": round((target_mid / quote["price"] - 1) * 100, 1) if target_mid else None,
        "target_last_month_avg": number((analysts.get("price_target_summary") or {}).get("lastMonthAvgPriceTarget")),
        "target_last_month_count": (analysts.get("price_target_summary") or {}).get("lastMonthCount"),
    }
    valuation_history = [v for v in prev.get("valuation_history", []) if v.get("date") != today]
    if valuation["forward_pe"]:
        valuation_history.append({"date": today, "price": quote["price"], "forward_pe": valuation["forward_pe"]})
    valuation_history = valuation_history[-400:]

    high52, low52 = number(metrics.get("52WeekHigh")), number(metrics.get("52WeekLow"))
    range_52w = None
    if high52 and low52 and high52 > low52:
        range_52w = {"high": high52, "low": low52, "position_pct": round((quote["price"] - low52) / (high52 - low52) * 100, 1),
                     "from_high_pct": round((quote["price"] / high52 - 1) * 100, 1)}

    # ----- news every run -----------------------------------------------------
    old_news = prev.get("news", {})
    items = dict(old_news.get("items", {}))
    titles = dict(old_news.get("story_titles", {}))
    query = company.get("news_query") or company.get("name") or ticker
    alpha_hours = ALPHA_NEWS_HOURS_PER_COMPANY * max(1, len(config["companies"]))
    feeds = {
        "finnhub": src.safe("finnhub.company_news", src.company_news, ticker, NEWS_DAYS, default=[]),
        "alphavantage": src.safe("alphavantage.news", src.alpha_news, ticker, default=[],
                                 min_hours=alpha_hours, cache_key=ticker),
        "yahoo": src.safe("rss.yahoo", src.yahoo_rss, ticker, default=[], cache_key=ticker),
        "google": src.safe("rss.google", src.google_news_rss, query, default=[], cache_key=ticker),
        "fmp": src.safe("fmp.news", src.fmp_news, ticker, default=[]),
    }
    fetched = merge_news(*feeds.values())
    known_keys = {headline_key(v.get("headline")) for v in items.values()}
    keywords = news_keywords(company)
    added = filtered = 0
    for item in fetched:
        key = headline_key(item["headline"])
        if item["id"] in items or key in known_keys:
            continue
        known_keys.add(key)
        if worth_sorting(item, keywords):
            items[item["id"]] = {**item, "first_seen": now_iso, "sorted": False, "sort_attempts": 0}
        else:
            # Kept on file so it is not checked again, but never sent to the AI.
            items[item["id"]] = {**item, "first_seen": now_iso, "sorted": True, "filtered": True, "relevant": False,
                                 "ai_summary": "", "impact": "neutral", "importance": "low", "topic": "other",
                                 "story": "filtered"}
            filtered += 1
        added += 1
    items = prune_items(items)
    print(f"{ticker} news fetched: " + ", ".join(f"{k} {len(v)}" for k, v in feeds.items())
          + f" | new {added}, {filtered} dropped by the free filter")

    pending = sorted((i for i in items.values() if not i.get("sorted")),
                     key=lambda i: i.get("published") or "", reverse=True)[:MAX_NEW_ITEMS_PER_RUN]
    newly_relevant, new_high, new_important = 0, False, False
    if pending and ai.can_spend("news", BUDGET_LIMITS["news"]):
        try:
            tags, new_titles = ai.sort_news(company, pending, {k: titles[k] for k in list(titles)[-80:]},
                                             config["ai"]["news_tokens"])
            titles.update(new_titles)
            for item in pending:
                tag = tags.get(item["id"])
                if tag:
                    items[item["id"]].update(tag, sorted=True)
                    if tag["relevant"]:
                        newly_relevant += 1
                        new_high = new_high or tag["importance"] == "high"
                        new_important = new_important or tag["importance"] in ("high", "medium")
                else:
                    items[item["id"]]["sort_attempts"] = item.get("sort_attempts", 0) + 1
            print(f"{ticker} sorted {len(tags)} new headlines ({newly_relevant} relevant)")
        except AIError as exc:
            errors.append(f"News sorting: {exc}")
            for item in pending:
                items[item["id"]]["sort_attempts"] = item.get("sort_attempts", 0) + 1
        for item in items.values():
            if not item.get("sorted") and item.get("sort_attempts", 0) >= MAX_SORT_ATTEMPTS:
                item.update(sorted=True, relevant=False, ai_summary="", impact="neutral",
                            importance="low", topic="other", story="unsorted")

    stories = build_stories(items, titles)
    used_keys = {i.get("story") for i in items.values()}
    titles = {k: v for k, v in titles.items() if k in used_keys}
    major_events = update_major_events(prev.get("major_events"), stories)
    market_news = src.safe("finnhub.market_news", src.market_news, default=None)
    if market_news is None:
        market_news = old_news.get("market", [])

    mood_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for s in stories:
        mood_counts[s["impact"]] = mood_counts.get(s["impact"], 0) + s["count"]
    scores = [i["sentiment_score"] for i in items.values()
              if i.get("relevant") and isinstance(i.get("sentiment_score"), (int, float))]
    sentiment_avg = round(sum(scores) / len(scores), 3) if scores else None
    by_source = {}
    for i in items.values():
        if i.get("relevant"):
            by_source[i.get("provider", "other")] = by_source.get(i.get("provider", "other"), 0) + 1

    # ----- AI brief and outlook ------------------------------------------------
    facts = {
        "market_open": market.get("is_open"),
        "price": quote["price"],
        "change_pct_today": quote["change_pct"],
        "day_low": quote["low"], "day_high": quote["high"],
        "versus_benchmarks": {s: {"benchmark_change_pct": b["change_pct"], "stock_minus_benchmark_pts": b["relative_pct"]}
                              for s, b in benchmarks.items()},
        "peers_change_pct": {s: p["change_pct"] for s, p in peers.items()},
        "range_52_week": range_52w,
        "forward_pe_next_12_months": valuation["forward_pe"],
        "forward_eps_next_12_months": valuation["forward_eps"],
        "next_fiscal_year_eps_consensus": valuation["next_year_eps"],
        "next_fiscal_year_end": valuation["next_fiscal_year_end"],
        "eps_revision_30d_pct": valuation["next_year_revision_30d_pct"],
        "analyst_price_target_consensus": target_mid,
        "upside_to_target_pct": valuation["target_upside_pct"],
        "price_target_note": "Many targets may predate recent price moves; compare with last month's average.",
        "price_target_last_month_avg": valuation["target_last_month_avg"],
        "next_earnings": earnings.get("next"),
    }
    top = story_digest(stories, 8)

    # The brief is only rewritten when something a reader would notice has changed.
    ai_conf = config["ai"]
    brief = prev.get("brief")
    brief_move = price_move_pct((brief or {}).get("price"), quote["price"])
    brief_reason = None
    if not brief:
        brief_reason = "first brief"
    elif new_important:
        brief_reason = "new important news"
    elif brief_move is not None and brief_move >= ai_conf["brief_price_move_pct"]:
        brief_reason = f"price moved {brief_move:.1f}% since the last brief"
    elif (hours_since(brief.get("written_at")) > BRIEF_MAX_AGE_HOURS
          and (newly_relevant > 0 or brief_move is None or brief_move >= 0.5)):
        brief_reason = f"refresh after {BRIEF_MAX_AGE_HOURS} hours"
    if brief_reason and ai.can_spend("brief", BUDGET_LIMITS["brief"]):
        try:
            brief = ai.write_brief(company, facts, top, ai_conf["brief_tokens"])
            brief.update(written_at=now_iso, price=quote["price"], reason=brief_reason)
            print(f"{ticker} brief ({brief_reason}): {brief['mood']}")
        except AIError as exc:
            errors.append(f"Brief: {exc}")
    elif not brief_reason:
        print(f"{ticker} brief kept, nothing important changed")

    # The outlook is rewritten once a day only if the news or price moved since the last one,
    # plus a limited number of early rewrites after high importance news.
    outlook = prev.get("outlook")
    outlooks_today = (prev.get("outlook_count") or {}).get(today, 0)
    written = (outlook or {}).get("written_at") or ""
    outlook_move = price_move_pct((outlook or {}).get("price"), quote["price"])
    outlook_reason, extra_run = None, False
    if not outlook:
        outlook_reason = "first outlook"
    elif written[:10] != today and (important_news_since(items, written) or outlook_move is None
                                    or outlook_move >= 3):
        outlook_reason = "daily update"
    elif written[:10] == today and new_high and outlooks_today < ai_conf["max_outlooks_per_day"]:
        outlook_reason, extra_run = "high importance news", True
    if outlook_reason and not ai.can_spend("outlook", BUDGET_LIMITS["outlook_extra" if extra_run else "outlook"]):
        outlook_reason = None
    if outlook_reason:
        extra = {
            "analyst_rating_trends": analysts.get("trends", [])[-3:],
            "analyst_consensus": analysts.get("consensus"),
            "price_target": analysts.get("price_target"),
            "recent_grade_changes": (analysts.get("latest_grades") or [])[:8],
            "earnings_surprises": earnings.get("history", [])[-4:],
            "upcoming_catalysts": catalysts[:8],
            "insider_activity_90d": insiders.get("summary"),
            "margins_growth": {k: metrics.get(k) for k in ("grossMarginTTM", "operatingMarginTTM",
                                                           "revenueGrowthTTMYoy", "epsGrowthTTMYoy")},
            "consensus_by_year": years,
        }
        try:
            outlook = ai.write_outlook(company, facts, story_digest(stories, 15), extra, ai_conf["outlook_tokens"])
            outlook.update(written_at=now_iso, price=quote["price"], reason=outlook_reason)
            outlooks_today += 1
            print(f"{ticker} outlook written ({outlook_reason})")
        except AIError as exc:
            errors.append(f"Outlook: {exc}")

    # ----- save -------------------------------------------------------------
    usage_day = prev.get("usage", {}).get("today", {})
    if usage_day.get("date") != today:
        usage_day = {"date": today, "runs": 0, "ai_tokens": 0}
    usage_day["runs"] += 1
    usage_day["filtered_headlines"] = usage_day.get("filtered_headlines", 0) + filtered
    usage_day["briefs_written"] = usage_day.get("briefs_written", 0) + int(bool(brief_reason) and
                                                                         (brief or {}).get("written_at") == now_iso)

    output = {
        "version": 2,
        "ticker": ticker,
        "name": company.get("name") or profile.get("name") or ticker,
        "updated_at": now_iso,
        "daily_done": today if daily_due else prev.get("daily_done"),
        "market": market,
        "profile": profile,
        "quote": quote,
        "range_52w": range_52w,
        "benchmarks": benchmarks,
        "benchmark_names": company.get("benchmark_names", {}),
        "peers": peers,
        "valuation": valuation,
        "valuation_history": valuation_history,
        "estimates": estimates,
        "eps_history": eps_history,
        "fundamentals": fundamentals,
        "analysts": analysts,
        "earnings": earnings,
        "insiders": insiders,
        "history": history,
        "catalysts": catalysts,
        "press_releases": press,
        "news": {
            "items": items,
            "story_titles": titles,
            "stories": stories,
            "mood_7d": mood_counts,
            "sentiment_avg": sentiment_avg,
            "by_provider": by_source,
            "market": market_news,
        },
        "major_events": major_events,
        "brief": brief,
        "outlook": outlook,
        "outlook_count": {today: outlooks_today},
        "drivers": company.get("drivers", []),
        "usage": {"today": usage_day},
        "errors": errors,
    }
    save_json(path, output)
    for error in errors:
        print(f"{ticker} problem: {error}")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_config():
    config = load_json(CONFIG_PATH, None)
    if not config or not config.get("companies"):
        sys.exit("config/companies.json is missing or has no companies.")
    ai_conf = config.get("ai") or {}
    fast = ai_conf.get("fast_model") or config.get("model") or "gemini-3.8-flash"
    config["ai"] = {
        "fast_model": fast,
        "deep_model": ai_conf.get("deep_model") or fast,
        "news_tokens": int(ai_conf.get("news_tokens") or 8000),
        "brief_tokens": max(4000, int(ai_conf.get("brief_tokens") or 0)),
        "outlook_tokens": int(ai_conf.get("outlook_tokens") or 8000),
        "reasoning_effort": ai_conf.get("reasoning_effort", "low") or None,
        "daily_token_budget": int(ai_conf.get("daily_token_budget") or 0) or None,
        "max_outlooks_per_day": int(ai_conf.get("max_outlooks_per_day") or 2),
        "brief_price_move_pct": float(ai_conf.get("brief_price_move_pct") or 2),
    }
    return config


def add_counts(total, new):
    for key, value in (new or {}).items():
        total[key] = total.get(key, 0) + value
    return total


def record_usage(usage, today, ai, budget):
    """Adds this run's AI use to data/_usage.json, one entry per day, the last 60 days kept."""
    days = [d for d in usage.get("days", []) if isinstance(d, dict) and d.get("date")]
    entry = next((d for d in days if d["date"] == today), None)
    if entry is None:
        entry = {"date": today, "runs": 0, "tokens": 0}
        days.append(entry)
    report = ai.report()
    entry["runs"] += 1
    entry["tokens"] += report["tokens"]
    for name in ("by_job", "calls_by_job", "thinking_by_job", "skipped_for_budget"):
        entry[name] = add_counts(entry.get(name, {}), report[name])
    days.sort(key=lambda d: d["date"])
    last7 = days[-7:]
    return {
        "budget_per_day": budget,
        "reasoning_effort": ai.reasoning_effort,
        "reasoning_effort_dropped_for": report["reasoning_effort_dropped_for"],
        "average_last_7_days": round(sum(d["tokens"] for d in last7) / len(last7)) if last7 else 0,
        "days": days[-USAGE_DAYS_KEPT:],
    }


def main():
    config = read_config()
    ai_conf = config["ai"]
    DATA_DIR.mkdir(exist_ok=True)
    capabilities = load_json(CAPABILITIES_PATH, {})
    usage = load_json(USAGE_PATH, {})
    today = today_iso()
    spent_today = next((d.get("tokens", 0) for d in usage.get("days", []) if d.get("date") == today), 0)

    alpha_key = os.environ.get("ALPHAVANTAGE_KEY", "").strip()
    if not alpha_key:
        print("note: no ALPHAVANTAGE_KEY set, so Alpha Vantage news and fund prices are skipped")
    src = Sources(get_key("FINNHUB_KEY"), get_key("FMP_KEY"), capabilities, alpha_key=alpha_key)
    ai = AI(get_key("HKBU_GENAI_API_KEY"), ai_conf["fast_model"], ai_conf["deep_model"],
            reasoning_effort=ai_conf["reasoning_effort"], daily_budget=ai_conf["daily_token_budget"],
            spent_today=spent_today)
    if ai_conf["daily_token_budget"]:
        print(f"AI budget: {spent_today:,} of {ai_conf['daily_token_budget']:,} tokens used today before this run")

    market = src.safe("finnhub.market_status", src.market_status, default={}) or {}
    print("US market open:", market.get("is_open"))

    failed = 0
    for company in config["companies"]:
        tokens_before = sum(ai.tokens.values())
        try:
            output = update_company(company, config, src, ai, market)
            output["usage"]["today"]["ai_tokens"] += sum(ai.tokens.values()) - tokens_before
            output["ai_budget"] = {"daily": ai_conf["daily_token_budget"], "used_today": ai.spent_today(),
                                   "used_pct": round(ai.used_pct(), 1) if ai_conf["daily_token_budget"] else None}
            save_json(DATA_DIR / f"{company['ticker']}.json", output)
        except Exception as exc:
            failed += 1
            print(f"{company['ticker']} failed: {type(exc).__name__}: {exc}")

    save_json(CAPABILITIES_PATH, src.capabilities)
    save_json(USAGE_PATH, record_usage(usage, today, ai, ai_conf["daily_token_budget"]))
    print("API calls this run:", dict(src.calls), "| AI tokens:", dict(ai.tokens))
    print("AI tokens by job:", dict(ai.job_tokens), "| thinking:", dict(ai.thinking_tokens),
          "| skipped for budget:", dict(ai.skipped) or "none")
    print("items returned:", src.sizes)
    for problem in src.problems:
        print("note:", problem)
    if failed == len(config["companies"]):
        sys.exit("Every company failed, so nothing new was saved.")


if __name__ == "__main__":
    main()

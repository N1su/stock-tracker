"""Data sources for the stock tracker.

Everything the tracker knows comes through this file. It talks to two free APIs:

  Finnhub  live quotes, company and market news, market status, company profile,
           key metrics and margins, analyst rating trends, earnings history,
           earnings calendar, insider trades, and a few endpoints that may be paid
  FMP      analyst estimates, daily price history, price targets, analyst grades,
           valuation ratios, quarterly income statements, and FMP news

Free plans change, so every call goes through Sources.safe(). If an endpoint
answers that it needs a paid plan, it is marked "plan_limited" and skipped for
seven days, which saves the daily call budget. Nothing here ever prints a key.

Run it directly to test every endpoint once:
    python scripts/sources.py NVDA
"""

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import requests

FINNHUB_BASE = "https://finnhub.io/api/v1"
FMP_BASE = "https://financialmodelingprep.com/stable"
TIMEOUT_SECONDS = 20
RECHECK_PLAN_LIMITED_DAYS = 7
PLAN_WORDS = ("premium", "subscription", "upgrade", "plan", "access", "restricted", "legacy")


class SourceError(Exception):
    """The call failed for an ordinary reason: network, bad key, bad answer."""


class PlanLimited(SourceError):
    """The endpoint exists but the free plan does not include it."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def utc_now():
    return datetime.now(timezone.utc)


def today_iso():
    return utc_now().date().isoformat()


def days_ago_iso(days):
    return (utc_now() - timedelta(days=days)).date().isoformat()


def days_ahead_iso(days):
    return (utc_now() + timedelta(days=days)).date().isoformat()


def from_unix(seconds):
    try:
        return datetime.fromtimestamp(int(seconds), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def number(value):
    """Return a float for numeric values and None for anything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pick(source, keys):
    """Copy the listed keys from a dict, using None where a key is missing."""
    source = source or {}
    return {key: source.get(key) for key in keys}


def clean_text(text, limit):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def news_id(url, headline):
    basis = (url or "").split("?")[0].lower() or (headline or "").lower()
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def headline_key(headline):
    return re.sub(r"[^a-z0-9 ]", "", (headline or "").lower())[:90]


def merge_news(*lists):
    """Combine news from several providers, drop exact repeats, newest first."""
    seen_ids, seen_titles, merged = set(), set(), []
    for items in lists:
        for item in items or []:
            title = headline_key(item.get("headline"))
            if not title or item["id"] in seen_ids or title in seen_titles:
                continue
            seen_ids.add(item["id"])
            seen_titles.add(title)
            merged.append(item)
    merged.sort(key=lambda item: item.get("published") or "", reverse=True)
    return merged


# ---------------------------------------------------------------------------
# The Sources class
# ---------------------------------------------------------------------------

class Sources:
    def __init__(self, finnhub_key, fmp_key, capabilities=None):
        self.finnhub_key = finnhub_key
        self.fmp_key = fmp_key
        self.capabilities = capabilities if capabilities is not None else {}
        self.calls = Counter()
        self.problems = []
        self.session = requests.Session()

    # ----- the safety wrapper ------------------------------------------------

    def safe(self, name, func, *args, default=None):
        """Run one endpoint. Never raises. Records what happened."""
        record = self.capabilities.get(name)
        if record and record.get("status") == "plan_limited":
            checked = record.get("checked", "")
            if checked and checked >= days_ago_iso(RECHECK_PLAN_LIMITED_DAYS):
                return default
        try:
            result = func(*args)
            self.capabilities[name] = {"status": "ok", "checked": today_iso()}
            return result
        except PlanLimited as exc:
            self.capabilities[name] = {"status": "plan_limited", "checked": today_iso(), "note": str(exc)[:160]}
            self.problems.append(f"{name}: not on the free plan")
            return default
        except Exception as exc:
            self.problems.append(f"{name}: {exc}")
            return default

    # ----- raw HTTP ---------------------------------------------------------------

    def _request(self, provider, url, params):
        self.calls[provider] += 1
        for attempt in range(2):
            try:
                response = self.session.get(url, params=params, timeout=TIMEOUT_SECONDS)
            except requests.RequestException as exc:
                raise SourceError(f"network problem ({type(exc).__name__})") from None
            if response.status_code == 429 and attempt == 0:
                wait = min(10.0, number(response.headers.get("Retry-After")) or 3.0)
                time.sleep(wait)
                self.calls[provider] += 1
                continue
            break

        text = response.text[:200]
        if response.status_code in (402, 403):
            raise PlanLimited(f"{response.status_code} {text}")
        if response.status_code == 401:
            raise SourceError("401 the key was refused")
        if response.status_code >= 400:
            raise SourceError(f"{response.status_code} {text}")
        try:
            data = response.json()
        except ValueError:
            raise SourceError("answer was not JSON") from None

        if isinstance(data, dict):
            message = data.get("error") or data.get("Error Message") or data.get("message")
            if message and len(data) <= 2:
                if any(word in str(message).lower() for word in PLAN_WORDS):
                    raise PlanLimited(str(message)[:200])
                raise SourceError(str(message)[:200])
        return data

    def _finnhub(self, path, **params):
        params["token"] = self.finnhub_key
        return self._request("finnhub", f"{FINNHUB_BASE}{path}", params)

    def _fmp(self, path, **params):
        params["apikey"] = self.fmp_key
        return self._request("fmp", f"{FMP_BASE}{path}", params)

    # =======================================================================
    # Finnhub
    # =======================================================================

    def quote(self, symbol):
        data = self._finnhub("/quote", symbol=symbol)
        if not number(data.get("c")):
            raise SourceError(f"no price for {symbol}")
        return {
            "price": number(data.get("c")),
            "change": number(data.get("d")),
            "change_pct": number(data.get("dp")) or 0.0,
            "high": number(data.get("h")),
            "low": number(data.get("l")),
            "open": number(data.get("o")),
            "prev_close": number(data.get("pc")),
            "time": from_unix(data.get("t")),
        }

    def market_status(self):
        data = self._finnhub("/stock/market-status", exchange="US")
        return {
            "is_open": bool(data.get("isOpen")),
            "session": data.get("session"),
            "holiday": data.get("holiday"),
            "checked": utc_now().isoformat(),
        }

    def profile(self, symbol):
        data = self._finnhub("/stock/profile2", symbol=symbol)
        if not data:
            raise SourceError(f"no profile for {symbol}")
        return {
            "name": data.get("name"),
            "logo": data.get("logo"),
            "industry": data.get("finnhubIndustry"),
            "exchange": data.get("exchange"),
            "market_cap_musd": number(data.get("marketCapitalization")),
            "shares_out_m": number(data.get("shareOutstanding")),
            "website": data.get("weburl"),
            "ipo": data.get("ipo"),
        }

    def company_news(self, symbol, days=7, limit=60):
        rows = self._finnhub("/company-news", symbol=symbol, **{"from": days_ago_iso(days), "to": today_iso()})
        if not isinstance(rows, list):
            raise SourceError("company news was not a list")
        items = []
        for row in rows[:limit]:
            url = row.get("url")
            headline = clean_text(row.get("headline"), 220)
            if not url or not headline:
                continue
            items.append({
                "id": news_id(url, headline),
                "headline": headline,
                "summary": clean_text(row.get("summary"), 400),
                "source": row.get("source") or "Unknown",
                "url": url,
                "image": row.get("image") or None,
                "published": from_unix(row.get("datetime")),
                "provider": "finnhub",
            })
        return items

    def market_news(self, limit=20):
        rows = self._finnhub("/news", category="general")
        if not isinstance(rows, list):
            raise SourceError("market news was not a list")
        items = []
        for row in rows[:limit]:
            url = row.get("url")
            headline = clean_text(row.get("headline"), 220)
            if not url or not headline:
                continue
            items.append({
                "id": news_id(url, headline),
                "headline": headline,
                "summary": clean_text(row.get("summary"), 300),
                "source": row.get("source") or "Unknown",
                "url": url,
                "published": from_unix(row.get("datetime")),
                "provider": "finnhub",
            })
        return items

    def basic_financials(self, symbol):
        data = self._finnhub("/stock/metric", symbol=symbol, metric="all")
        metric = data.get("metric") or {}
        if not metric:
            raise SourceError("no metrics returned")
        keys = [
            "52WeekHigh", "52WeekLow", "52WeekHighDate", "52WeekLowDate",
            "52WeekPriceReturnDaily", "13WeekPriceReturnDaily", "26WeekPriceReturnDaily",
            "5DayPriceReturnDaily", "monthToDatePriceReturnDaily", "yearToDatePriceReturnDaily",
            "beta", "marketCapitalization", "peTTM", "peBasicExclExtraTTM", "pegTTM", "psTTM", "pbAnnual",
            "grossMarginTTM", "operatingMarginTTM", "netProfitMarginTTM",
            "revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy", "epsGrowthTTMYoy", "epsGrowthQuarterlyYoy",
            "roeTTM", "roiTTM", "currentDividendYieldTTM",
            "10DayAverageTradingVolume", "3MonthAverageTradingVolume",
        ]
        quarterly = (data.get("series") or {}).get("quarterly") or {}
        series = {}
        for name in ("grossMargin", "operatingMargin", "netMargin", "eps", "salesPerShare", "fcfMargin"):
            points = quarterly.get(name) or []
            series[name] = [
                {"period": p.get("period"), "value": number(p.get("v"))}
                for p in points[:12] if p.get("period")
            ][::-1]
        return {"metrics": pick(metric, keys), "quarterly": series}

    def recommendation_trends(self, symbol):
        rows = self._finnhub("/stock/recommendation", symbol=symbol)
        if not isinstance(rows, list):
            raise SourceError("recommendations were not a list")
        return [
            pick(row, ["period", "strongBuy", "buy", "hold", "sell", "strongSell"])
            for row in rows[:6]
        ][::-1]

    def earnings_history(self, symbol):
        rows = self._finnhub("/stock/earnings", symbol=symbol)
        if not isinstance(rows, list):
            raise SourceError("earnings history was not a list")
        return [
            pick(row, ["period", "quarter", "year", "actual", "estimate", "surprise", "surprisePercent"])
            for row in rows[:8]
        ][::-1]

    def earnings_calendar(self, symbol, days_ahead=120):
        data = self._finnhub("/calendar/earnings", symbol=symbol,
                             **{"from": days_ago_iso(1), "to": days_ahead_iso(days_ahead)})
        rows = data.get("earningsCalendar") or []
        return [
            pick(row, ["date", "hour", "quarter", "year", "epsEstimate", "revenueEstimate"])
            for row in sorted(rows, key=lambda r: r.get("date") or "")
        ]

    def insider_transactions(self, symbol, days=180):
        data = self._finnhub("/stock/insider-transactions", symbol=symbol, **{"from": days_ago_iso(days)})
        rows = data.get("data") or []
        rows.sort(key=lambda r: r.get("transactionDate") or "", reverse=True)
        return [
            pick(row, ["name", "transactionDate", "filingDate", "transactionCode",
                       "change", "share", "transactionPrice"])
            for row in rows[:40]
        ]

    def insider_sentiment(self, symbol):
        data = self._finnhub("/stock/insider-sentiment", symbol=symbol,
                             **{"from": days_ago_iso(365), "to": today_iso()})
        rows = data.get("data") or []
        return [pick(row, ["year", "month", "change", "mspr"]) for row in rows[-12:]]

    def finnhub_price_target(self, symbol):
        data = self._finnhub("/stock/price-target", symbol=symbol)
        if not data.get("targetMean"):
            raise SourceError("no price target data")
        return pick(data, ["targetHigh", "targetLow", "targetMean", "targetMedian", "lastUpdated"])

    def upgrades_downgrades(self, symbol, days=90):
        rows = self._finnhub("/stock/upgrade-downgrade", symbol=symbol, **{"from": days_ago_iso(days)})
        if not isinstance(rows, list):
            raise SourceError("upgrades were not a list")
        return [
            {**pick(row, ["company", "fromGrade", "toGrade", "action"]), "date": from_unix(row.get("gradeTime"))}
            for row in rows[:20]
        ]

    # =======================================================================
    # FMP
    # =======================================================================

    def analyst_estimates(self, symbol):
        rows = self._fmp("/analyst-estimates", symbol=symbol, period="annual", limit=10)
        if not isinstance(rows, list) or not rows:
            raise SourceError("no estimates returned")
        today = today_iso()
        future = sorted((r for r in rows if (r.get("date") or "") > today), key=lambda r: r["date"])
        return [
            {
                "fiscal_year_end": r["date"],
                "eps_avg": number(r.get("epsAvg")),
                "eps_low": number(r.get("epsLow")),
                "eps_high": number(r.get("epsHigh")),
                "revenue_avg": number(r.get("revenueAvg")),
                "revenue_low": number(r.get("revenueLow")),
                "revenue_high": number(r.get("revenueHigh")),
                "net_income_avg": number(r.get("netIncomeAvg")),
                "analysts_eps": r.get("numAnalystsEps"),
                "analysts_revenue": r.get("numAnalystsRevenue"),
            }
            for r in future[:4]
        ]

    def price_history(self, symbol, days=400):
        rows = self._fmp("/historical-price-eod/full", symbol=symbol,
                         **{"from": days_ago_iso(days), "to": today_iso()})
        if isinstance(rows, dict):
            rows = rows.get("historical") or []
        if not isinstance(rows, list) or not rows:
            raise SourceError("no price history returned")
        points = [
            {
                "date": r.get("date"),
                "open": number(r.get("open")),
                "high": number(r.get("high")),
                "low": number(r.get("low")),
                "close": number(r.get("close")),
                "volume": number(r.get("volume")),
            }
            for r in rows if r.get("date") and number(r.get("close"))
        ]
        points.sort(key=lambda p: p["date"])
        return points

    def price_target_consensus(self, symbol):
        rows = self._fmp("/price-target-consensus", symbol=symbol)
        row = rows[0] if isinstance(rows, list) and rows else rows
        if not isinstance(row, dict) or not row.get("targetConsensus"):
            raise SourceError("no price target consensus")
        return pick(row, ["targetHigh", "targetLow", "targetConsensus", "targetMedian"])

    def price_target_summary(self, symbol):
        rows = self._fmp("/price-target-summary", symbol=symbol)
        row = rows[0] if isinstance(rows, list) and rows else rows
        if not isinstance(row, dict):
            raise SourceError("no price target summary")
        return pick(row, ["lastMonthCount", "lastMonthAvgPriceTarget", "lastQuarterCount",
                          "lastQuarterAvgPriceTarget", "lastYearCount", "lastYearAvgPriceTarget"])

    def grades_consensus(self, symbol):
        rows = self._fmp("/grades-consensus", symbol=symbol)
        row = rows[0] if isinstance(rows, list) and rows else rows
        if not isinstance(row, dict):
            raise SourceError("no grades consensus")
        return pick(row, ["strongBuy", "buy", "hold", "sell", "strongSell", "consensus"])

    def latest_grades(self, symbol, limit=15):
        rows = self._fmp("/grades", symbol=symbol)
        if not isinstance(rows, list):
            raise SourceError("grades were not a list")
        rows.sort(key=lambda r: r.get("date") or "", reverse=True)
        return [pick(r, ["date", "gradingCompany", "previousGrade", "newGrade", "action"]) for r in rows[:limit]]

    def ratios_ttm(self, symbol):
        rows = self._fmp("/ratios-ttm", symbol=symbol)
        row = rows[0] if isinstance(rows, list) and rows else rows
        if not isinstance(row, dict) or not row:
            raise SourceError("no ratios returned")
        return {k: number(v) for k, v in row.items() if number(v) is not None}

    def key_metrics_ttm(self, symbol):
        rows = self._fmp("/key-metrics-ttm", symbol=symbol)
        row = rows[0] if isinstance(rows, list) and rows else rows
        if not isinstance(row, dict) or not row:
            raise SourceError("no key metrics returned")
        return {k: number(v) for k, v in row.items() if number(v) is not None}

    def quarterly_income(self, symbol, limit=8):
        rows = self._fmp("/income-statement", symbol=symbol, period="quarter", limit=limit)
        if not isinstance(rows, list) or not rows:
            raise SourceError("no quarterly income statements")
        out = [
            {
                "date": r.get("date"),
                "period": f"{r.get('fiscalYear') or r.get('calendarYear') or ''} {r.get('period') or ''}".strip(),
                "revenue": number(r.get("revenue")),
                "gross_profit": number(r.get("grossProfit")),
                "operating_income": number(r.get("operatingIncome")),
                "net_income": number(r.get("netIncome")),
                "eps": number(r.get("epsDiluted") or r.get("epsdiluted") or r.get("eps")),
            }
            for r in rows
        ]
        out.sort(key=lambda r: r.get("date") or "")
        return out

    def fmp_news(self, symbol, limit=30):
        rows = self._fmp("/news/stock", symbols=symbol, limit=limit)
        if not isinstance(rows, list):
            raise SourceError("FMP news was not a list")
        items = []
        for row in rows:
            url = row.get("url")
            headline = clean_text(row.get("title"), 220)
            if not url or not headline:
                continue
            published = row.get("publishedDate")
            if published and "T" not in published:
                published = published.replace(" ", "T") + "+00:00"
            items.append({
                "id": news_id(url, headline),
                "headline": headline,
                "summary": clean_text(row.get("text"), 400),
                "source": row.get("site") or row.get("publisher") or "Unknown",
                "url": url,
                "image": row.get("image") or None,
                "published": published,
                "provider": "fmp",
            })
        return items

    def press_releases(self, symbol, limit=10):
        rows = self._fmp("/news/press-releases", symbols=symbol, limit=limit)
        if not isinstance(rows, list):
            raise SourceError("press releases were not a list")
        return [
            {
                "date": row.get("publishedDate"),
                "title": clean_text(row.get("title"), 220),
                "url": row.get("url"),
            }
            for row in rows if row.get("title")
        ]


# ---------------------------------------------------------------------------
# Self test: python scripts/sources.py NVDA
# ---------------------------------------------------------------------------

def self_test(symbol):
    finnhub_key = os.environ.get("FINNHUB_KEY", "").strip()
    fmp_key = os.environ.get("FMP_KEY", "").strip()
    if not finnhub_key or not fmp_key:
        sys.exit("Set FINNHUB_KEY and FMP_KEY first.")

    src = Sources(finnhub_key, fmp_key)
    checks = [
        ("finnhub.quote", src.quote, symbol),
        ("finnhub.market_status", src.market_status),
        ("finnhub.profile", src.profile, symbol),
        ("finnhub.company_news", src.company_news, symbol),
        ("finnhub.market_news", src.market_news),
        ("finnhub.basic_financials", src.basic_financials, symbol),
        ("finnhub.recommendation_trends", src.recommendation_trends, symbol),
        ("finnhub.earnings_history", src.earnings_history, symbol),
        ("finnhub.earnings_calendar", src.earnings_calendar, symbol),
        ("finnhub.insider_transactions", src.insider_transactions, symbol),
        ("finnhub.insider_sentiment", src.insider_sentiment, symbol),
        ("finnhub.price_target", src.finnhub_price_target, symbol),
        ("finnhub.upgrades_downgrades", src.upgrades_downgrades, symbol),
        ("fmp.analyst_estimates", src.analyst_estimates, symbol),
        ("fmp.price_history", src.price_history, symbol),
        ("fmp.price_target_consensus", src.price_target_consensus, symbol),
        ("fmp.price_target_summary", src.price_target_summary, symbol),
        ("fmp.grades_consensus", src.grades_consensus, symbol),
        ("fmp.latest_grades", src.latest_grades, symbol),
        ("fmp.ratios_ttm", src.ratios_ttm, symbol),
        ("fmp.key_metrics_ttm", src.key_metrics_ttm, symbol),
        ("fmp.quarterly_income", src.quarterly_income, symbol),
        ("fmp.news", src.fmp_news, symbol),
        ("fmp.press_releases", src.press_releases, symbol),
    ]
    for name, func, *args in checks:
        result = src.safe(name, func, *args)
        status = src.capabilities.get(name, {}).get("status", "error")
        size = len(result) if isinstance(result, (list, dict)) else (0 if result is None else 1)
        print(f"{status:13} {name:34} items: {size}")
    print("calls used:", dict(src.calls))
    for problem in src.problems:
        print("problem:", problem)
    print(json.dumps(src.capabilities, indent=2))


if __name__ == "__main__":
    self_test(sys.argv[1] if len(sys.argv) > 1 else "NVDA")

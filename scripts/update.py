"""Stock tracker update script.

For every company in config/companies.json this script:
1. gets live quotes from Finnhub (the company, its benchmarks and its peers)
2. gets analyst estimates from FMP once a day
3. works out forward EPS, forward P/E and the 30 day estimate revision
4. asks the HKBU GenAI API for a three line brief
5. saves everything to data/<TICKER>.json

Keys are read from environment variables, which GitHub fills from Secrets.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "companies.json"
DATA_DIR = ROOT / "data"

FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
FMP_ESTIMATES_URL = "https://financialmodelingprep.com/stable/analyst-estimates"
HKBU_CHAT_URL = "https://genai.hkbu.edu.hk/api/v1/chat/completions"

TIMEOUT_SECONDS = 30
HKBU_TIMEOUT_SECONDS = 120


def get_key(name):
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"Missing environment variable {name}. Check the GitHub secret name.")
    return value


def check_response(response, label):
    # Report the status and message without printing the URL, which holds the key.
    if response.status_code >= 400:
        raise RuntimeError(f"{label} returned {response.status_code}: {response.text[:300]}")


def load_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def get_quote(symbol, key):
    response = requests.get(
        FINNHUB_QUOTE_URL,
        params={"symbol": symbol, "token": key},
        timeout=TIMEOUT_SECONDS,
    )
    check_response(response, f"Finnhub quote {symbol}")
    data = response.json()
    if not data.get("c"):
        raise ValueError(f"No price returned for {symbol}")
    return {
        "price": data["c"],
        "change": data.get("d"),
        "change_pct": data.get("dp") or 0.0,
        "high": data.get("h"),
        "low": data.get("l"),
        "open": data.get("o"),
        "prev_close": data.get("pc"),
        "time": datetime.fromtimestamp(data.get("t", 0), tz=timezone.utc).isoformat(),
    }


def get_estimates(symbol, key, today):
    response = requests.get(
        FMP_ESTIMATES_URL,
        params={"symbol": symbol, "period": "annual", "limit": 10, "apikey": key},
        timeout=TIMEOUT_SECONDS,
    )
    check_response(response, f"FMP estimates {symbol}")
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No estimates returned for {symbol}: {str(rows)[:200]}")
    future = sorted(
        (row for row in rows if row.get("date", "") > today),
        key=lambda row: row["date"],
    )
    return [
        {
            "fiscal_year_end": row["date"],
            "eps_avg": row.get("epsAvg"),
            "eps_low": row.get("epsLow"),
            "eps_high": row.get("epsHigh"),
            "revenue_avg": row.get("revenueAvg"),
            "analysts_eps": row.get("numAnalystsEps"),
        }
        for row in future[:3]
    ]


def forward_eps(years, today):
    # Next twelve months EPS: blend this fiscal year and next by time remaining.
    if len(years) < 2 or years[0]["eps_avg"] is None or years[1]["eps_avg"] is None:
        return None
    year_end = datetime.fromisoformat(years[0]["fiscal_year_end"]).date()
    days_left = (year_end - datetime.fromisoformat(today).date()).days
    weight = max(0.0, min(1.0, days_left / 365))
    return round(weight * years[0]["eps_avg"] + (1 - weight) * years[1]["eps_avg"], 2)


def update_history(history, today, years):
    history = [entry for entry in history if entry.get("date") != today]
    history.append({"date": today, "eps": {y["fiscal_year_end"]: y["eps_avg"] for y in years}})
    return history[-90:]


def revision_30d(history, fiscal_year_end, today):
    cutoff = (datetime.fromisoformat(today) - timedelta(days=30)).date().isoformat()
    points = [
        entry["eps"][fiscal_year_end]
        for entry in history
        if entry.get("date", "") >= cutoff and entry.get("eps", {}).get(fiscal_year_end)
    ]
    if len(points) < 2:
        return None
    return round((points[-1] - points[0]) / points[0] * 100, 2)


def write_brief(company, facts, key, model, max_tokens):
    system = (
        "You are an equity analyst writing for a beginner investor. "
        "Use only the numbers you are given and never invent news or reasons. "
        "Do not give buy or sell advice. Use plain English and no dashes as punctuation."
    )
    user = (
        f"Here is today's data for {company['name']} ({company['ticker']}):\n"
        f"{json.dumps(facts, indent=2)}\n\n"
        "Write exactly three short lines, one sentence each, with no bullets or numbering:\n"
        "Line 1: what the stock did today.\n"
        "Line 2: how that compares with the market benchmarks and peers.\n"
        "Line 3: what the valuation and estimate numbers mean for a long term holder."
    )
    response = requests.post(
        HKBU_CHAT_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=HKBU_TIMEOUT_SECONDS,
    )
    check_response(response, "HKBU GenAI")
    payload = response.json()
    text = (payload["choices"][0]["message"].get("content") or "").strip()
    if not text:
        raise ValueError("HKBU GenAI returned an empty reply")
    lines = [line.strip(" -*").strip() for line in text.splitlines() if line.strip()]
    return {
        "lines": lines[:3],
        "model": model,
        "total_tokens": payload.get("usage", {}).get("total_tokens"),
    }


def update_company(company, keys, config):
    ticker = company["ticker"]
    path = DATA_DIR / f"{ticker}.json"
    previous = load_json(path)
    today = datetime.now(timezone.utc).date().isoformat()
    errors = []

    quote = get_quote(ticker, keys["finnhub"])
    print(f"{ticker} price: {quote['price']} ({quote['change_pct']}%)")

    benchmarks = {}
    for symbol in company.get("benchmarks", []):
        try:
            bench = get_quote(symbol, keys["finnhub"])
            benchmarks[symbol] = {
                "change_pct": bench["change_pct"],
                "relative_pct": round(quote["change_pct"] - bench["change_pct"], 2),
            }
        except Exception as exc:
            errors.append(f"Benchmark {symbol}: {exc}")

    peers = {}
    for symbol in company.get("peers", []):
        try:
            peer = get_quote(symbol, keys["finnhub"])
            peers[symbol] = {"price": peer["price"], "change_pct": peer["change_pct"]}
        except Exception as exc:
            errors.append(f"Peer {symbol}: {exc}")

    estimates = previous.get("estimates", {})
    history = previous.get("eps_history", [])
    if estimates.get("fetched_on") != today or not estimates.get("years"):
        try:
            years = get_estimates(ticker, keys["fmp"], today)
            estimates = {"fetched_on": today, "years": years}
            history = update_history(history, today, years)
            print(f"{ticker} estimates: refreshed")
        except Exception as exc:
            errors.append(f"Estimates: {exc}")

    years = estimates.get("years", [])
    eps_ntm = forward_eps(years, today)
    next_year = years[1]["fiscal_year_end"] if len(years) > 1 else None
    valuation = {
        "forward_eps": eps_ntm,
        "forward_pe": round(quote["price"] / eps_ntm, 1) if eps_ntm and eps_ntm > 0 else None,
        "next_fiscal_year_end": next_year,
        "next_year_revision_30d_pct": revision_30d(history, next_year, today) if next_year else None,
    }

    facts = {
        "price": quote["price"],
        "change_pct_today": quote["change_pct"],
        "day_low": quote["low"],
        "day_high": quote["high"],
        "versus_benchmarks": benchmarks,
        "peers_change_pct": {symbol: p["change_pct"] for symbol, p in peers.items()},
        "valuation": valuation,
    }

    brief = previous.get("brief")
    try:
        brief = write_brief(company, facts, keys["hkbu"], config["model"], config["max_tokens"])
        print(f"{ticker} HKBU brief: OK ({brief['total_tokens']} tokens)")
    except Exception as exc:
        errors.append(f"HKBU brief: {exc}")

    output = {
        "ticker": ticker,
        "name": company["name"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "quote": quote,
        "benchmarks": benchmarks,
        "peers": peers,
        "valuation": valuation,
        "estimates": estimates,
        "eps_history": history,
        "brief": brief,
        "errors": errors,
    }
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for error in errors:
        print(f"{ticker} problem: {error}")


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    keys = {
        "finnhub": get_key("FINNHUB_KEY"),
        "fmp": get_key("FMP_KEY"),
        "hkbu": get_key("HKBU_GENAI_API_KEY"),
    }
    DATA_DIR.mkdir(exist_ok=True)

    failed = 0
    for company in config["companies"]:
        try:
            update_company(company, keys, config)
        except Exception as exc:
            failed += 1
            print(f"{company['ticker']} failed: {exc}")

    if failed == len(config["companies"]):
        sys.exit("Every company failed, so nothing was saved.")


if __name__ == "__main__":
    main()

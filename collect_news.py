"""
Historical News Headline Collector  (2021 – 2026)
==================================================
Collects headlines from multiple free APIs and appends to all_news_collected.csv.

Sources
-------
  A — Alpha Vantage  : 25 req/day free, returns pre-scored sentiment
  B — Finnhub        : 60 req/min free, ~1 year history
  C — Alpaca         : 200 req/min free, history back to 2015  ← best for 2021-2025
  D — Tiingo         : 1000 req/day free, multi-year history

Usage
-----
    python collect_news.py                    # all tickers
    python collect_news.py AAPL               # single ticker
    python collect_news.py AAPL MSFT NVDA     # multiple tickers

Keys needed in .env
-------------------
    AV_API_KEY=...
    FINNHUB_API_KEY=...
    ALPACA_API_KEY=...
    ALPACA_API_SECRET=...
    TIINGO_API_KEY=...
"""

import os, sys, ssl, time, json
import urllib3, requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Keys ──────────────────────────────────────────────────────────────────────
AV_API_KEY        = os.getenv("AV_API_KEY",        "YOUR_ALPHAVANTAGE_KEY")
FINNHUB_API_KEY   = os.getenv("FINNHUB_API_KEY",   "YOUR_FINNHUB_KEY")
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",     "YOUR_ALPACA_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET",  "YOUR_ALPACA_SECRET")
TIINGO_API_KEY    = os.getenv("TIINGO_API_KEY",     "YOUR_TIINGO_KEY")

# ── Config ────────────────────────────────────────────────────────────────────
ALL_TICKERS = ["AAPL", "AMD", "AMZN", "AVGO", "GOOGL",
               "META", "MSFT", "NVDA", "ORCL", "TSLA"]
START_DATE  = "2021-01-01"
END_DATE    = "2026-06-22"
OUTPUT_FILE = Path("all_news_collected.csv")
CACHE_DIR   = Path("news_collected")
CACHE_DIR.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.verify = False

# Shared output columns
COLS = ["ticker", "date", "source", "headline", "summary", "url",
        "av_sentiment_label", "av_sentiment_score", "av_relevance_score",
        "sentiment_label", "sentiment_score", "gdelt_tone", "source_api"]


# ── Date range helpers ────────────────────────────────────────────────────────

def _monthly_ranges(start, end):
    cur    = datetime.strptime(start, "%Y-%m-%d").replace(day=1)
    end_dt = datetime.strptime(end,   "%Y-%m-%d")
    while cur <= end_dt:
        if cur.month == 12:
            m_end = cur.replace(year=cur.year+1, month=1, day=1) - timedelta(days=1)
        else:
            m_end = cur.replace(month=cur.month+1, day=1) - timedelta(days=1)
        m_end = min(m_end, end_dt)
        yield cur.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d")
        cur = m_end + timedelta(days=1)

def _quarter_ranges(start, end):
    cur    = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end,   "%Y-%m-%d")
    while cur <= end_dt:
        q_end = min(cur + timedelta(days=89), end_dt)
        yield cur.strftime("%Y%m%dT%H%M"), q_end.strftime("%Y%m%dT%H%M")
        cur = q_end + timedelta(days=1)


def _empty_row(ticker, date, headline, summary, url, label="", score=None,
               relevance=None, source="", source_api=""):
    return {
        "ticker": ticker, "date": date, "source": source,
        "headline": headline, "summary": summary, "url": url,
        "av_sentiment_label": label, "av_sentiment_score": score,
        "av_relevance_score": relevance, "sentiment_label": "",
        "sentiment_score": None, "gdelt_tone": None,
        "source_api": source_api,
    }


# ── Source A — Alpha Vantage ──────────────────────────────────────────────────

def fetch_alphavantage(ticker: str) -> pd.DataFrame:
    if AV_API_KEY == "YOUR_ALPHAVANTAGE_KEY":
        print("  [AV] Skipping — no AV_API_KEY in .env")
        return pd.DataFrame()

    rows     = []
    quarters = list(_quarter_ranges(START_DATE, END_DATE))
    print(f"  [AV] {ticker}: {len(quarters)} quarterly windows ...")

    for i, (t_from, t_to) in enumerate(quarters):
        cache = CACHE_DIR / f"{ticker}_{t_from[:6]}_av.json"
        if cache.exists():
            data = json.load(open(cache, encoding="utf-8"))
        else:
            try:
                resp = SESSION.get("https://www.alphavantage.co/query", params={
                    "function": "NEWS_SENTIMENT", "tickers": ticker,
                    "time_from": t_from, "time_to": t_to,
                    "limit": 1000, "sort": "EARLIEST", "apikey": AV_API_KEY,
                }, timeout=30)
                data = resp.json()
                if "Information" in data or "Note" in data:
                    print(f"  [AV] Rate limit hit — re-run tomorrow.")
                    break
                json.dump(data, open(cache, "w"))
            except Exception as e:
                print(f"    AV error {t_from}: {e}")
                continue
            time.sleep(13)   # ~4.6 req/min — safe under 25/day

        for art in data.get("feed", []):
            pub  = art.get("time_published", "")
            date = f"{pub[:4]}-{pub[4:6]}-{pub[6:8]}" if len(pub) >= 8 else ""
            ts   = next((x for x in art.get("ticker_sentiment", [])
                         if x.get("ticker") == ticker), {})
            rows.append(_empty_row(
                ticker    = ticker,
                date      = date,
                headline  = art.get("title", ""),
                summary   = art.get("summary", ""),
                url       = art.get("url", ""),
                label     = art.get("overall_sentiment_label", ""),
                score     = float(art.get("overall_sentiment_score", 0) or 0),
                relevance = float(ts.get("relevance_score", 0) or 0),
                source    = art.get("source", ""),
                source_api= "alphavantage",
            ))

        print(f"    Q{i+1}/{len(quarters)} ({t_from[:6]}): "
              f"{len(data.get('feed', []))} articles | total: {len(rows)}")

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Source B — Finnhub ────────────────────────────────────────────────────────

def fetch_finnhub(ticker: str) -> pd.DataFrame:
    if FINNHUB_API_KEY == "YOUR_FINNHUB_KEY":
        print("  [Finnhub] Skipping — no FINNHUB_API_KEY in .env")
        return pd.DataFrame()

    rows   = []
    months = list(_monthly_ranges(START_DATE, END_DATE))
    print(f"  [Finnhub] {ticker}: {len(months)} monthly windows ...")

    for i, (m_from, m_to) in enumerate(months):
        cache = CACHE_DIR / f"{ticker}_{m_from[:7].replace('-','')}_finnhub.json"
        if cache.exists():
            articles = json.load(open(cache, encoding="utf-8"))
        else:
            try:
                resp     = SESSION.get("https://finnhub.io/api/v1/company-news",
                    params={"symbol": ticker, "from": m_from,
                            "to": m_to, "token": FINNHUB_API_KEY}, timeout=20)
                articles = resp.json()
                if isinstance(articles, dict) and "error" in articles:
                    articles = []
                json.dump(articles, open(cache, "w", encoding="utf-8"),
                          ensure_ascii=False)
            except Exception as e:
                print(f"    Finnhub error {m_from}: {e}")
                articles = []
            time.sleep(1)

        for art in (articles if isinstance(articles, list) else []):
            ts   = art.get("datetime", 0)
            date = (datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
                    if ts else "")
            rows.append(_empty_row(
                ticker    = ticker,
                date      = date,
                headline  = art.get("headline", ""),
                summary   = art.get("summary", ""),
                url       = art.get("url", ""),
                source    = art.get("source", ""),
                source_api= "finnhub",
            ))

        if (i + 1) % 12 == 0:
            print(f"    {i+1}/{len(months)} months | total: {len(rows)}")

    print(f"  [Finnhub] {len(rows)} articles for {ticker}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Source C — Alpaca ─────────────────────────────────────────────────────────

def fetch_alpaca(ticker: str) -> pd.DataFrame:
    if ALPACA_API_KEY == "YOUR_ALPACA_KEY":
        print("  [Alpaca] Skipping — no ALPACA_API_KEY in .env")
        return pd.DataFrame()

    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY,
               "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    rows    = []
    months  = list(_monthly_ranges(START_DATE, END_DATE))
    print(f"  [Alpaca] {ticker}: {len(months)} monthly windows ...")

    for i, (m_from, m_to) in enumerate(months):
        cache = CACHE_DIR / f"{ticker}_{m_from[:7].replace('-','')}_alpaca.json"
        if cache.exists():
            articles = json.load(open(cache, encoding="utf-8"))
        else:
            articles   = []
            page_token = None
            while True:
                params = {"symbols": ticker,
                          "start": f"{m_from}T00:00:00Z",
                          "end":   f"{m_to}T23:59:59Z",
                          "limit": 50, "sort": "asc"}
                if page_token:
                    params["page_token"] = page_token
                try:
                    resp = SESSION.get(
                        "https://data.alpaca.markets/v1beta1/news",
                        headers=headers, params=params, timeout=20)
                    data = resp.json()
                    articles.extend(data.get("news", []))
                    page_token = data.get("next_page_token")
                    if not page_token:
                        break
                    time.sleep(0.3)
                except Exception as e:
                    print(f"    Alpaca error {m_from}: {e}")
                    break
            json.dump(articles, open(cache, "w", encoding="utf-8"),
                      ensure_ascii=False)
            time.sleep(0.5)

        for art in articles:
            pub  = art.get("created_at", "")
            rows.append(_empty_row(
                ticker    = ticker,
                date      = pub[:10] if pub else "",
                headline  = art.get("headline", ""),
                summary   = art.get("summary", ""),
                url       = art.get("url", ""),
                source    = art.get("source", ""),
                source_api= "alpaca",
            ))

        if (i + 1) % 12 == 0:
            print(f"    {i+1}/{len(months)} months | total: {len(rows)}")

    print(f"  [Alpaca] {len(rows)} articles for {ticker}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Source D — Tiingo ─────────────────────────────────────────────────────────

def fetch_tiingo(ticker: str) -> pd.DataFrame:
    if TIINGO_API_KEY == "YOUR_TIINGO_KEY":
        print("  [Tiingo] Skipping — no TIINGO_API_KEY in .env")
        return pd.DataFrame()

    rows   = []
    months = list(_monthly_ranges(START_DATE, END_DATE))
    print(f"  [Tiingo] {ticker}: {len(months)} monthly windows ...")

    for i, (m_from, m_to) in enumerate(months):
        cache = CACHE_DIR / f"{ticker}_{m_from[:7].replace('-','')}_tiingo.json"
        if cache.exists():
            articles = json.load(open(cache, encoding="utf-8"))
        else:
            articles = []
            offset   = 0
            while True:
                try:
                    resp  = SESSION.get("https://api.tiingo.com/tiingo/news",
                        params={"tickers": ticker.lower(), "startDate": m_from,
                                "endDate": m_to, "limit": 1000, "offset": offset,
                                "token": TIINGO_API_KEY}, timeout=20)
                    batch = resp.json()
                    if not isinstance(batch, list) or not batch:
                        break
                    articles.extend(batch)
                    if len(batch) < 1000:
                        break
                    offset += 1000
                    time.sleep(0.5)
                except Exception as e:
                    print(f"    Tiingo error {m_from}: {e}")
                    break
            json.dump(articles, open(cache, "w", encoding="utf-8"),
                      ensure_ascii=False)
            time.sleep(0.5)

        for art in articles:
            pub  = art.get("publishedDate", "")
            rows.append(_empty_row(
                ticker    = ticker,
                date      = pub[:10] if pub else "",
                headline  = art.get("title", ""),
                summary   = art.get("description", ""),
                url       = art.get("url", ""),
                source    = art.get("source", ""),
                source_api= "tiingo",
            ))

        if (i + 1) % 12 == 0:
            print(f"    {i+1}/{len(months)} months | total: {len(rows)}")

    print(f"  [Tiingo] {len(rows)} articles for {ticker}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Collect one ticker ────────────────────────────────────────────────────────

def collect_ticker(ticker: str) -> pd.DataFrame:
    print(f"\n{'='*55}\n  {ticker}\n{'='*55}")
    frames = []
    for fn, label in [(fetch_alphavantage, "AV"),
                      (fetch_finnhub,      "Finnhub"),
                      (fetch_alpaca,       "Alpaca"),
                      (fetch_tiingo,       "Tiingo")]:
        df = fn(ticker)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(subset=["ticker", "headline"])
    print(f"  Deduplicated: {before} -> {len(out)}")
    return out


# ── Main — append to all_news_collected.csv ───────────────────────────────────

def main():
    tickers = [t.upper() for t in sys.argv[1:]] if sys.argv[1:] else ALL_TICKERS

    print("=" * 55)
    print(f"  News Collector  {START_DATE} -> {END_DATE}")
    print(f"  Tickers : {tickers}")
    active = []
    if AV_API_KEY       != "YOUR_ALPHAVANTAGE_KEY": active.append("AlphaVantage")
    if FINNHUB_API_KEY  != "YOUR_FINNHUB_KEY":      active.append("Finnhub")
    if ALPACA_API_KEY   != "YOUR_ALPACA_KEY":        active.append("Alpaca")
    if TIINGO_API_KEY   != "YOUR_TIINGO_KEY":        active.append("Tiingo")
    print(f"  Active APIs: {active if active else 'NONE — check .env'}")
    print("=" * 55)

    frames = []
    for ticker in tickers:
        df = collect_ticker(ticker)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("\nNo data collected. Check your API keys in .env")
        return

    new_data = pd.concat(frames, ignore_index=True)
    new_data["date"] = pd.to_datetime(new_data["date"],
                                      errors="coerce").dt.strftime("%Y-%m-%d")
    new_data = new_data.dropna(subset=["date", "headline"])
    new_data = new_data[new_data["headline"].str.strip() != ""]

    # ── Append to existing all_news_collected.csv ─────────────────────────────
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, encoding="utf-8",
                               encoding_errors="replace")
        print(f"\nExisting file: {len(existing)} rows")
        combined = pd.concat([existing, new_data], ignore_index=True)
        before   = len(combined)
        combined = combined.drop_duplicates(subset=["ticker", "headline"])
        combined.to_csv(OUTPUT_FILE, index=False)
        print(f"Appended {len(new_data)} new rows")
        print(f"After dedup: {before} -> {len(combined)} rows")
    else:
        new_data.to_csv(OUTPUT_FILE, index=False)
        print(f"\nCreated {OUTPUT_FILE}: {len(new_data)} rows")

    # ── Summary ───────────────────────────────────────────────────────────────
    final = pd.read_csv(OUTPUT_FILE)
    print(f"\n{'='*55}")
    print(f"  all_news_collected.csv  —  {len(final)} total rows")
    print(f"  Sources: {final['source_api'].value_counts().to_dict()}")
    dates = pd.to_datetime(final['date'], errors='coerce').dropna()
    print(f"  Date range: {dates.min().date()} -> {dates.max().date()}")
    print(f"\n  NEXT: python run_finbert.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()

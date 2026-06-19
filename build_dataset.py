"""
Dataset Builder
===============
Downloads historical OHLCV data from yfinance for all tickers
and saves a clean master_dataset.csv ready for train_model.py.

No API keys required — uses yfinance (free).

Usage
-----
    python build_dataset.py                        # all 10 tickers, 5 years
    python build_dataset.py AAPL TSLA              # specific tickers
    python build_dataset.py --start 2020-01-01     # custom start date
    python build_dataset.py --out my_dataset.csv   # custom output file
"""

import sys
import time
import warnings
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

ALL_TICKERS = ["AAPL", "AMD", "AMZN", "AVGO", "GOOGL",
               "META", "MSFT", "NVDA", "ORCL", "TSLA"]

DEFAULT_START  = "2018-01-01"
DEFAULT_OUTPUT = "master_dataset.csv"


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.Ticker(ticker).history(
        start=start, end=end,
        interval="1d", auto_adjust=True
    )
    if raw.empty:
        print(f"  [{ticker}] WARNING: no data returned")
        return pd.DataFrame()

    if hasattr(raw.index, "tz") and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)

    df = pd.DataFrame({
        "date":   raw.index.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "open":   raw["Open"].round(6),
        "high":   raw["High"].round(6),
        "low":    raw["Low"].round(6),
        "close":  raw["Close"].round(6),
        "volume": raw["Volume"].astype(int),
    })

    # Drop rows with any zero/null prices (bad data)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[(df["close"] > 0) & (df["volume"] > 0)]

    return df.reset_index(drop=True)


def main():
    args       = sys.argv[1:]
    start_date = DEFAULT_START
    output     = DEFAULT_OUTPUT

    if "--start" in args:
        idx        = args.index("--start")
        start_date = args[idx + 1]
        args       = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    if "--out" in args:
        idx    = args.index("--out")
        output = args[idx + 1]
        args   = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    tickers = [t.upper() for t in args] if args else ALL_TICKERS
    end_date = datetime.today().strftime("%Y-%m-%d")

    print("=" * 55)
    print("  Dataset Builder (yfinance)")
    print(f"  Tickers : {tickers}")
    print(f"  Range   : {start_date} -> {end_date}")
    print(f"  Output  : {output}")
    print("=" * 55)

    frames = []
    for ticker in tickers:
        print(f"  Downloading {ticker} ...", end=" ", flush=True)
        df = download_ticker(ticker, start_date, end_date)
        if df.empty:
            continue
        frames.append(df)
        print(f"{len(df)} rows  ({df['date'].iloc[0]} -> {df['date'].iloc[-1]})")
        time.sleep(0.3)   # be polite to yfinance

    if not frames:
        print("No data downloaded. Check ticker symbols or internet connection.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Sanity checks
    dups = combined.duplicated(subset=["date", "ticker"]).sum()
    if dups:
        print(f"  WARNING: {dups} duplicate (date, ticker) rows removed")
        combined = combined.drop_duplicates(subset=["date", "ticker"])

    combined.to_csv(output, index=False)

    print()
    print("=" * 55)
    print(f"  Saved {len(combined)} rows -> {output}")
    print()
    print("  Rows per ticker:")
    for ticker, grp in combined.groupby("ticker"):
        print(f"    {ticker:<6}  {len(grp):>5} rows   "
              f"{grp['date'].iloc[0]} -> {grp['date'].iloc[-1]}")
    print()
    print("  Next step:")
    print(f"    python train_model.py ALL --csv {output}")
    print("=" * 55)


if __name__ == "__main__":
    main()

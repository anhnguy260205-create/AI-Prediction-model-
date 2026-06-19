"""
News Dataset Cleaner
====================
Reads all_news_combined.csv, applies data-quality filters, and writes
all_news_combined_clean.csv ready to pass into run_finbert.py.

Usage
-----
    python clean_news.py                          # default I/O
    python clean_news.py my_input.csv             # custom input
    python clean_news.py my_input.csv my_out.csv  # custom I/O
"""

import sys
import pandas as pd

INPUT_FILE  = sys.argv[1] if len(sys.argv) > 1 else "all_news_collected.csv"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "all_news_collected_clean.csv"

HEADLINE_MIN_LEN = 15   # chars — below this it's a ticker symbol or fragment
SUMMARY_MIN_LEN  = 20   # chars of real content (after stripping whitespace/newlines)

# Keywords that must appear in headline OR summary to keep the article
COMPANY_KEYWORDS = {
    "AAPL":  ["apple", "aapl", "tim cook", "iphone", "ipad", "imac", "ios",
               "wwdc", "app store", "airpods", "macbook", "siri"],
    "MSFT":  ["microsoft", "msft", "satya nadella", "azure", "windows",
               "copilot", "teams", "xbox", "bing", "linkedin"],
    "NVDA":  ["nvidia", "nvda", "jensen huang", "gpu", "cuda",
               "h100", "h200", "blackwell", "geforce", "rtx"],
    "GOOGL": ["google", "googl", "alphabet", "sundar pichai", "youtube",
               "gemini", "android", "chrome", "waymo", "deepmind"],
    "AMZN":  ["amazon", "amzn", "aws", "andy jassy", "prime",
               "alexa", "kindle", "whole foods", "twitch"],
    "META":  ["meta ", "meta's", "facebook", "instagram", "whatsapp",
               "mark zuckerberg", "zuckerberg", "threads", "ray-ban",
               "oculus", "llama"],
    "TSLA":  ["tesla", "tsla", "elon musk", "elon", "cybertruck",
               "model 3", "model s", "model x", "model y",
               "powerwall", "supercharger", "full self-driving", "fsd"],
    "AMD":   ["amd", "advanced micro devices", "lisa su",
               "ryzen", "radeon", "epyc", "instinct"],
    "AVGO":  ["broadcom", "avgo", "hock tan", "vmware", "brocade"],
    "ORCL":  ["oracle", "orcl", "larry ellison", "cloud@customer",
               "netsuite", "cerner"],
}


def _clean_text(s: pd.Series) -> pd.Series:
    return s.str.strip().str.replace(r"\s+", " ", regex=True)


def _is_empty(s: pd.Series) -> pd.Series:
    return s.fillna("").str.strip() == ""


def report(label: str, before: int, after: int) -> None:
    dropped = before - after
    pct = dropped / before * 100 if before else 0
    print(f"  {label:<48} -{dropped:>5}  ({pct:.1f}%)  -> {after} rows remain")


def is_relevant(df: pd.DataFrame) -> pd.Series:
    """True where headline or summary contains at least one company keyword."""
    result = pd.Series(False, index=df.index)
    text = (df["headline"].fillna("") + " " + df["summary"].fillna("")).str.lower()
    for ticker, keywords in COMPANY_KEYWORDS.items():
        mask = df["ticker"] == ticker
        for kw in keywords:
            result |= mask & text.str.contains(kw, regex=False)
    return result


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    print(f"\n  Starting rows : {n0}")
    print()

    # ── 1. Drop fully-null columns ────────────────────────────────────────────
    fully_null = [c for c in df.columns if df[c].isna().all()]
    if fully_null:
        print(f"  Dropping fully-null columns: {fully_null}")
        df = df.drop(columns=fully_null)

    # ── 2. Normalise text fields ──────────────────────────────────────────────
    df["headline"] = _clean_text(df["headline"].astype(str))
    df["summary"]  = _clean_text(df["summary"].fillna(""))
    df.loc[_is_empty(df["summary"]), "summary"] = None

    # ── 3. Deduplicate by URL ─────────────────────────────────────────────────
    before = len(df)
    df = df.sort_values("date").drop_duplicates(subset=["url"], keep="first")
    report("Duplicate URLs removed", before, len(df))

    # ── 4. Deduplicate by (ticker, headline) ─────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "headline"], keep="first")
    report("Duplicate (ticker, headline) removed", before, len(df))

    # ── 5. Remove cross-ticker duplicates (same headline, different tickers) ──
    # These are generic market articles incorrectly tagged to multiple stocks.
    before = len(df)
    df = df.drop_duplicates(subset=["headline"], keep=False)
    report("Cross-ticker duplicate headlines removed", before, len(df))

    # ── 6. Drop rows with headline too short to carry meaning ─────────────────
    before = len(df)
    df = df[df["headline"].str.len() >= HEADLINE_MIN_LEN]
    report(f"Headlines < {HEADLINE_MIN_LEN} chars removed", before, len(df))

    # ── 7. Drop rows where summary is missing or too short ───────────────────
    before = len(df)
    df = df[df["summary"].notna() & (df["summary"].str.len() >= SUMMARY_MIN_LEN)]
    report("Missing/short summaries removed", before, len(df))

    # ── 8. Drop junk summaries (bare URL, byline fragment) ───────────────────
    before = len(df)
    junk = (
        df["summary"].str.match(r"^https?://", na=False)
        | df["summary"].str.match(r"^-\w", na=False)
        | df["summary"].str.match(r"^\w+\.\w+\s*$", na=False)
    )
    df = df[~junk]
    report("Junk summaries (bare URL / byline) removed", before, len(df))

    # ── 9. Remove off-topic articles ─────────────────────────────────────────
    # Keep only rows where headline or summary mentions the company by name.
    before = len(df)
    df = df[is_relevant(df)]
    report("Off-topic articles removed (no company mention)", before, len(df))

    # ── 10. Parse date and sort ───────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    before = len(df)
    df = df.dropna(subset=["date"])
    if len(df) < before:
        report("Unparseable dates removed", before, len(df))

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print()
    print(f"  Final rows    : {len(df)}  (removed {n0 - len(df)} total, "
          f"{(n0 - len(df)) / n0 * 100:.1f}%)")
    return df


def main():
    print("=" * 60)
    print(f"  News Dataset Cleaner")
    print(f"  Input  : {INPUT_FILE}")
    print(f"  Output : {OUTPUT_FILE}")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    df = clean(df)

    print()
    print("  Rows per ticker after cleaning:")
    for ticker, count in df["ticker"].value_counts().sort_index().items():
        print(f"    {ticker:<6} {count}")

    df.to_csv(OUTPUT_FILE, index=False)
    print()
    print(f"  Saved -> {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()

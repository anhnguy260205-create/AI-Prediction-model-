"""
FinBERT Sentiment Scorer
========================
Reads headlines from all_news_collected_clean.csv,
scores each one with ProsusAI/finbert,
aggregates per (date, ticker), and
merges the result into master_dataset.csv.

Usage
-----
    python run_finbert.py                        # uses all_news_collected_clean.csv
    python run_finbert.py all_news_combined.csv  # use existing file
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import pipeline

INPUT_NEWS   = sys.argv[1] if len(sys.argv) > 1 else "all_news_collected_clean.csv"
MASTER       = "master_dataset.csv"
OUTPUT_NEWS  = "all_news_finbert.csv"
OUTPUT_MASTER = "master_dataset_with_sentiment.csv"
BATCH_SIZE   = 32
CACHE_PATH   = Path("news_collected/finbert_scores_cache.csv")


# ── Load model once ───────────────────────────────────────────────────────────

def load_finbert():
    print("[FinBERT] Loading ProsusAI/finbert ...")
    pipe = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,
        truncation=True,
        max_length=128,
        device=-1,         # CPU; change to 0 for GPU
    )
    print("[FinBERT] Model loaded.")
    return pipe


# ── Score headlines ───────────────────────────────────────────────────────────

def score_headlines(pipe, texts: list[str]) -> pd.DataFrame:
    """
    Run FinBERT on a list of headline strings.
    Returns DataFrame with columns: pos, neg, neu, score (pos-neg).
    """
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        preds = pipe(batch)
        for pred in preds:
            d = {r["label"]: r["score"] for r in pred}
            results.append({
                "pos":   d.get("positive", 0.0),
                "neg":   d.get("negative", 0.0),
                "neu":   d.get("neutral",  0.0),
                "score": d.get("positive", 0.0) - d.get("negative", 0.0),
            })
        if (i // BATCH_SIZE + 1) % 10 == 0:
            done = min(i + BATCH_SIZE, len(texts))
            print(f"  Scored {done}/{len(texts)} headlines ...")
    return pd.DataFrame(results)


# ── Aggregate per (date, ticker) ──────────────────────────────────────────────

def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by date + ticker, compute mean sentiment scores.
    Returns one row per (date, ticker).
    Lags are NOT computed here — they are computed after merging with the
    master dataset so that zero-news trading days are correctly counted as 0.
    """
    agg = (
        df.groupby(["date", "ticker"])
        .agg(
            finbert_score_mean=("finbert_score",  "mean"),
            finbert_pos_mean  =("finbert_pos",    "mean"),
            finbert_neg_mean  =("finbert_neg",    "mean"),
            news_count        =("headline",       "count"),
        )
        .reset_index()
    )
    return agg


# ── Merge into master_dataset ─────────────────────────────────────────────────

def merge_into_master(agg: pd.DataFrame) -> pd.DataFrame:
    master = pd.read_csv(MASTER)
    master["date"] = pd.to_datetime(master["date"]).dt.strftime("%Y-%m-%d")
    agg["date"]    = pd.to_datetime(agg["date"]).dt.strftime("%Y-%m-%d")

    # Drop old empty sentiment columns from master
    drop_cols = [c for c in master.columns
                 if c in {"finbert_avg_score","gdelt_avg_tone",
                          "av_avg_score","av_bullish_ratio",
                          "av_bearish_ratio","av_neutral_ratio",
                          "sentiment_lag1","sentiment_lag2"}]
    master = master.drop(columns=drop_cols, errors="ignore")

    merged = master.merge(agg, on=["date","ticker"], how="left")

    # Fill trading days with no news as zero BEFORE computing lags,
    # so zero-news days contribute 0 to the lag (not a skip).
    base_sent_cols = ["finbert_score_mean","finbert_pos_mean",
                      "finbert_neg_mean","news_count"]
    merged[base_sent_cols] = merged[base_sent_cols].fillna(0.0)

    # Compute lags over the full trading-day calendar (zeros included)
    merged = merged.sort_values(["ticker", "date"])
    merged["sentiment_lag1"] = (
        merged.groupby("ticker")["finbert_score_mean"].shift(1).fillna(0.0)
    )
    merged["sentiment_lag2"] = (
        merged.groupby("ticker")["finbert_score_mean"].shift(2).fillna(0.0)
    )

    filled = (merged["finbert_score_mean"] != 0).sum()
    total  = len(merged)
    print(f"\n[Merge] Rows with real sentiment : {filled}/{total} ({filled/total*100:.1f}%)")
    print(f"[Merge] Rows with zero sentiment : {total-filled}/{total}")
    return merged


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load news
    if not Path(INPUT_NEWS).exists():
        print(f"ERROR: {INPUT_NEWS} not found.")
        print("Run  python collect_news.py  first.")
        return

    print(f"[Step 1] Loading headlines from {INPUT_NEWS} ...")
    news = pd.read_csv(INPUT_NEWS, encoding="utf-8", encoding_errors="replace")
    print(f"         {len(news)} rows loaded")

    # Filter rows where headline is not empty
    news = news[news["headline"].notna() & (news["headline"].str.strip() != "")]
    print(f"         {len(news)} rows with valid headlines")

    # Check cache — skip already-scored rows
    if CACHE_PATH.exists():
        cache = pd.read_csv(CACHE_PATH)
        scored_idx = set(cache["original_idx"].astype(int))
        news_to_score = news[~news.index.isin(scored_idx)].copy()
        print(f"[Step 2] {len(scored_idx)} headlines already in cache — "
              f"scoring {len(news_to_score)} new ones ...")
    else:
        cache = pd.DataFrame()
        news_to_score = news.copy()
        print(f"[Step 2] Scoring all {len(news_to_score)} headlines with FinBERT ...")

    # Score
    if not news_to_score.empty:
        pipe    = load_finbert()
        texts   = news_to_score["headline"].tolist()
        scores  = score_headlines(pipe, texts)
        scores.index = news_to_score.index
        scores["original_idx"] = news_to_score.index

        new_cache = pd.concat([cache, scores], ignore_index=True)
        CACHE_PATH.parent.mkdir(exist_ok=True)
        new_cache.to_csv(CACHE_PATH, index=False)
        print(f"         Cache updated -> {CACHE_PATH}")
        full_scores = new_cache
    else:
        full_scores = cache

    # Attach scores back to news DataFrame
    full_scores = full_scores.set_index("original_idx")
    news["finbert_score"] = full_scores["score"].reindex(news.index).values
    news["finbert_pos"]   = full_scores["pos"].reindex(news.index).values
    news["finbert_neg"]   = full_scores["neg"].reindex(news.index).values

    # Save scored news
    news.to_csv(OUTPUT_NEWS, index=False)
    print(f"\n[Step 3] Scored news saved -> {OUTPUT_NEWS}")

    # Aggregate
    print(f"[Step 4] Aggregating per (date, ticker) ...")
    agg = aggregate_sentiment(news)
    print(f"         {len(agg)} unique (date, ticker) combinations")
    print(f"         Date range: {agg['date'].min()} -> {agg['date'].max()}")

    # Merge into master
    print(f"[Step 5] Merging into {MASTER} ...")
    if not Path(MASTER).exists():
        print(f"         WARNING: {MASTER} not found — saving aggregated scores only.")
        agg.to_csv("sentiment_scores.csv", index=False)
        print(f"         Sentiment scores saved -> sentiment_scores.csv")
        return

    final = merge_into_master(agg)
    final.to_csv(OUTPUT_MASTER, index=False)
    print(f"[Step 5] Final dataset saved -> {OUTPUT_MASTER}")

    # Report
    print(f"\n{'='*55}")
    print(f"  SENTIMENT COVERAGE REPORT")
    print(f"{'='*55}")
    for ticker in sorted(final["ticker"].unique()):
        sub   = final[final["ticker"] == ticker]
        filled = (sub["finbert_score_mean"] != 0).sum()
        print(f"  {ticker:<6}  {filled:>4}/{len(sub)} days with real sentiment "
              f"({filled/len(sub)*100:.1f}%)")
    print(f"{'='*55}")
    print(f"\n  NEXT STEP: train model on enriched dataset")
    print(f"    python train_model.py AAPL 5y --csv {OUTPUT_MASTER}")


if __name__ == "__main__":
    main()

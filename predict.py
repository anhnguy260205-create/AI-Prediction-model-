"""
Stock Direction Predictor
=========================
Loads a saved XGBoost model and predicts next-day price direction
for one or more tickers using the latest market data from yfinance.

Usage
-----
    python predict.py                                        # AAPL, latest model
    python predict.py TSLA                                   # single ticker
    python predict.py AAPL TSLA NVDA                         # multiple tickers
    python predict.py TSLA --model xgb_tsla_auc0.5602.json  # specific model
    python predict.py ALL                                    # all 10 tickers
"""

import sys
import warnings
import glob
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb

warnings.filterwarnings("ignore")

ALL_TICKERS = ["AAPL", "AMD", "AMZN", "AVGO", "GOOGL",
               "META", "MSFT", "NVDA", "ORCL", "TSLA"]

LOOKBACK_DAYS = 90   # days of history to download — enough for MA20 + lags
FINBERT_NEWS  = "all_news_finbert.csv"


# ── Feature builders (mirrors train_model.py) ─────────────────────────────────

def _rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _macd_diff(close, fast=12, slow=26, sig=9):
    m = (close.ewm(span=fast, adjust=False).mean()
         - close.ewm(span=slow, adjust=False).mean())
    return m - m.ewm(span=sig, adjust=False).mean()


def _stoch(high, low, close, k=14, d=3):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    sk = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    return sk, sk.rolling(d).mean()


def _bb(close, period=20, n=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + n * std
    lower = mid - n * std
    return (2 * n * std) / mid.replace(0, np.nan), \
           (close - lower) / (upper - lower).replace(0, np.nan)


def _atr_pct(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(),
                    (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / close


def _load_sentiment_for_ticker(ticker: str) -> dict:
    """Return most recent FinBERT scores for a ticker from all_news_finbert.csv."""
    if not Path(FINBERT_NEWS).exists():
        return {}
    try:
        news = pd.read_csv(FINBERT_NEWS,
                           usecols=["ticker", "date", "finbert_score",
                                    "finbert_pos", "finbert_neg"])
        news = news[news["ticker"] == ticker].copy()
        news["date"] = pd.to_datetime(news["date"]).dt.date
        daily = (
            news.groupby("date")
            .agg(finbert_score_mean=("finbert_score", "mean"),
                 finbert_pos_mean  =("finbert_pos",   "mean"),
                 finbert_neg_mean  =("finbert_neg",   "mean"),
                 news_count        =("finbert_score", "count"))
            .reset_index()
            .sort_values("date")
        )
        if daily.empty:
            return {}
        return {
            "finbert_score_mean": daily.iloc[-1]["finbert_score_mean"],
            "finbert_pos_mean":   daily.iloc[-1]["finbert_pos_mean"],
            "finbert_neg_mean":   daily.iloc[-1]["finbert_neg_mean"],
            "news_count":         daily.iloc[-1]["news_count"],
            "sentiment_lag1":     daily.iloc[-2]["finbert_score_mean"] if len(daily) >= 2 else 0.0,
            "sentiment_lag2":     daily.iloc[-3]["finbert_score_mean"] if len(daily) >= 3 else 0.0,
        }
    except Exception:
        return {}


def _build_all_features(df: pd.DataFrame, spy: pd.DataFrame,
                        vix: pd.DataFrame, sentiment: dict | None = None) -> pd.DataFrame:
    """Build every possible feature — model will select what it needs."""
    close  = df["Close"]; high = df["High"]
    low    = df["Low"];   volume = df["Volume"]
    out    = pd.DataFrame(index=df.index)
    ma5    = close.rolling(5).mean()
    ma20   = close.rolling(20).mean()
    bb_w, bb_p = _bb(close)
    sk, sd = _stoch(high, low, close)
    rsi_s  = _rsi(close)

    out["return_1d"]   = close.pct_change(1)
    out["return_5d"]   = close.pct_change(5)
    out["return_10d"]  = close.pct_change(10)
    out["return_20d"]  = close.pct_change(20)
    out["price_to_ma5"]  = close / ma5.replace(0, np.nan)
    out["price_to_ma20"] = close / ma20.replace(0, np.nan)
    out["ma5_to_ma20"]   = ma5 / ma20.replace(0, np.nan)
    out["rsi"]       = rsi_s
    out["macd_diff"] = _macd_diff(close)
    out["stoch_k"]   = sk
    out["stoch_d"]   = sd
    out["bb_width"]  = bb_w
    out["bb_pct"]    = bb_p
    out["atr_pct"]   = _atr_pct(high, low, close)
    out["volume_ratio"] = volume / volume.rolling(20).mean()
    out["return_lag1"]  = out["return_1d"].shift(1)
    out["return_lag2"]  = out["return_1d"].shift(2)
    out["return_lag3"]  = out["return_1d"].shift(3)
    out["rsi_lag1"]  = rsi_s.shift(1)
    out["rsi_lag2"]  = rsi_s.shift(2)
    out["rsi_lag3"]  = rsi_s.shift(3)
    out["day_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 5)
    out["day_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 5)

    # Sentiment — from all_news_finbert.csv if available, else zero
    sent = sentiment or {}
    out["finbert_score_mean"] = sent.get("finbert_score_mean", 0.0)
    out["finbert_pos_mean"]   = sent.get("finbert_pos_mean",   0.0)
    out["finbert_neg_mean"]   = sent.get("finbert_neg_mean",   0.0)
    out["news_count"]         = sent.get("news_count",         0.0)
    out["sentiment_lag1"]     = sent.get("sentiment_lag1",     0.0)
    out["sentiment_lag2"]     = sent.get("sentiment_lag2",     0.0)
    out["has_sentiment"]      = 1.0 if sent else 0.0

    # Market features
    for col in ["spy_return_1d", "spy_return_5d", "spy_ma_ratio"]:
        if col in spy.columns:
            out[col] = spy[col].reindex(out.index, method="ffill", fill_value=0.0)
    for col in ["vix_level", "vix_change", "vix_ma_ratio"]:
        if col in vix.columns:
            out[col] = vix[col].reindex(out.index, method="ffill", fill_value=0.0)

    return out


# ── Data download ─────────────────────────────────────────────────────────────

def _download(ticker: str, days: int) -> pd.DataFrame:
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    raw   = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if hasattr(raw.index, "tz") and raw.index.tz:
        raw.index = raw.index.tz_localize(None)
    return raw[["Open", "High", "Low", "Close", "Volume"]]


def _download_market(days: int):
    spy_raw = yf.Ticker("SPY").history(
        start=(datetime.today()-timedelta(days=days)).strftime("%Y-%m-%d"),
        auto_adjust=True)
    vix_raw = yf.Ticker("^VIX").history(
        start=(datetime.today()-timedelta(days=days)).strftime("%Y-%m-%d"),
        auto_adjust=True)
    for raw in [spy_raw, vix_raw]:
        if hasattr(raw.index, "tz") and raw.index.tz:
            raw.index = raw.index.tz_localize(None)

    spy = pd.DataFrame(index=spy_raw.index)
    spy["spy_return_1d"] = spy_raw["Close"].pct_change(1)
    spy["spy_return_5d"] = spy_raw["Close"].pct_change(5)
    spy["spy_ma_ratio"]  = spy_raw["Close"] / spy_raw["Close"].rolling(20).mean()

    vix = pd.DataFrame(index=vix_raw.index)
    vix["vix_level"]    = vix_raw["Close"]
    vix["vix_change"]   = vix_raw["Close"].pct_change(1)
    vix["vix_ma_ratio"] = vix_raw["Close"] / vix_raw["Close"].rolling(20).mean()

    return spy, vix


# ── Find latest model ─────────────────────────────────────────────────────────

def _find_latest_model() -> str:
    models = glob.glob("xgb_*.json")
    models = [m for m in models if m != "xgb_stock_model.json"]
    if not models:
        raise FileNotFoundError("No xgb_*.json model files found. Run train_model.py first.")
    return max(models, key=os.path.getmtime)


# ── Predict ───────────────────────────────────────────────────────────────────

def predict(ticker: str, model: xgb.XGBClassifier,
            spy: pd.DataFrame, vix: pd.DataFrame) -> dict:
    df      = _download(ticker, LOOKBACK_DAYS)
    sent    = _load_sentiment_for_ticker(ticker)
    feats   = _build_all_features(df, spy, vix, sentiment=sent)
    needed  = list(model.feature_names_in_)

    # Select only features the model was trained on
    missing = [f for f in needed if f not in feats.columns]
    for f in missing:
        feats[f] = 0.0

    row = feats[needed].dropna().iloc[-1:]
    if row.empty:
        return {"ticker": ticker, "error": "Not enough data to compute features"}

    proba     = float(model.predict_proba(row)[0, 1])
    direction = "UP" if proba >= 0.5 else "DOWN"
    confidence = abs(proba - 0.5) * 200   # 0–100% scale

    last_close = float(df["Close"].iloc[-1])
    last_date  = df.index[-1].date()

    return {
        "ticker":     ticker,
        "as_of":      str(last_close),
        "last_date":  str(last_date),
        "direction":  direction,
        "prob_up":    round(proba, 4),
        "prob_down":  round(1 - proba, 4),
        "confidence": round(confidence, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args       = sys.argv[1:]
    model_file = None

    if "--model" in args:
        idx        = args.index("--model")
        model_file = args[idx + 1]
        args       = [a for i, a in enumerate(args) if i not in (idx, idx+1)]

    if not args or args[0].upper() == "ALL":
        tickers = ALL_TICKERS
    else:
        tickers = [t.upper() for t in args]

    if model_file is None:
        model_file = _find_latest_model()

    print("=" * 55)
    print(f"  Stock Direction Predictor")
    print(f"  Model   : {model_file}")
    print(f"  Tickers : {tickers}")
    print(f"  Run at  : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    model = xgb.XGBClassifier()
    model.load_model(model_file)
    print(f"  Features used: {len(model.feature_names_in_)}")
    print()

    print("  Downloading market data (SPY + VIX) ...")
    spy, vix = _download_market(LOOKBACK_DAYS)

    print()
    print(f"  {'Ticker':<8} {'Last Close':>10}  {'Date':<12} "
          f"{'Signal':<6} {'P(UP)':>6}  {'P(DOWN)':>7}  {'Confidence':>10}")
    print(f"  {'-'*8} {'-'*10}  {'-'*12} {'-'*6} {'-'*6}  {'-'*7}  {'-'*10}")

    results = []
    for ticker in tickers:
        try:
            r = predict(ticker, model, spy, vix)
            if "error" in r:
                print(f"  {ticker:<8}  ERROR: {r['error']}")
                continue
            arrow = "^" if r["direction"] == "UP" else "v"
            print(f"  {r['ticker']:<8} {float(r['as_of']):>10.2f}  {r['last_date']:<12} "
                  f"{arrow} {r['direction']:<4} {r['prob_up']:>6.1%}  "
                  f"{r['prob_down']:>7.1%}  {r['confidence']:>9.1f}%")
            results.append(r)
        except Exception as e:
            print(f"  {ticker:<8}  ERROR: {e}")

    print()
    if results:
        up   = [r for r in results if r["direction"] == "UP"]
        down = [r for r in results if r["direction"] == "DOWN"]
        print(f"  Summary: {len(up)} UP  |  {len(down)} DOWN  |  "
              f"avg confidence {sum(r['confidence'] for r in results)/len(results):.1f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()

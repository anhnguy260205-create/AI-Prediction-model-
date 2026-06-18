"""
XGBoost Stock Direction Classifier
====================================
Loads OHLCV + pre-scored FinBERT sentiment from master_dataset_with_sentiment.csv,
engineers technical indicators, and trains an XGBoost binary classifier to predict
next-day price direction (up=1 / down=0).

Usage
-----
    python train_model.py                              # AAPL only
    python train_model.py TSLA                         # single ticker
    python train_model.py ALL                          # all 10 tickers combined
    python train_model.py AAPL TSLA NVDA               # specific tickers combined
    python train_model.py ALL --no-sentiment           # technical + SPY only
    python train_model.py ALL --start-date 2025-06-01  # trim to sentiment window
    python train_model.py ALL --csv other_master.csv   # custom input file
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score,
    f1_score, precision_score, recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

MASTER_CSV = "master_dataset_with_sentiment.csv"

ALL_TICKERS = ["AAPL", "AMD", "AMZN", "AVGO", "GOOGL",
               "META", "MSFT", "NVDA", "ORCL", "TSLA"]

# Full feature set matching xgb_stock_model baseline + extra
TECHNICAL_FEATURES = [
    "return_1d", "return_2d", "return_5d",
    "price_to_ma5", "price_to_ma20", "ma5_to_ma20",
    "rsi", "macd_diff", "stoch_k", "stoch_d",
    "bb_width", "bb_pct", "atr_pct", "volume_ratio",
    "return_lag1", "return_lag2", "return_lag3",
    "rsi_lag1", "rsi_lag2", "rsi_lag3",
]

SENTIMENT_FEATURES = [
    "finbert_score_mean", "finbert_pos_mean", "finbert_neg_mean",
    "news_count", "sentiment_lag1", "sentiment_lag2",
    "has_sentiment",   # binary flag: 1 if real FinBERT scores exist, 0 if zeroed
]

MARKET_FEATURES = [
    "spy_return_1d",
    "spy_return_5d",
    "spy_ma_ratio",
]

ALL_FEATURES = TECHNICAL_FEATURES + SENTIMENT_FEATURES + MARKET_FEATURES

SPY_CACHE = Path(".cache/spy_features.csv")


# ── SPY market features ───────────────────────────────────────────────────────

def load_spy_features() -> pd.DataFrame:
    if SPY_CACHE.exists():
        print(f"         SPY features loaded from cache ({SPY_CACHE})")
        return pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True)

    print(f"         Downloading SPY data from yfinance ...")
    SPY_CACHE.parent.mkdir(exist_ok=True)
    raw = yf.Ticker("SPY").history(period="10y", interval="1d", auto_adjust=True)
    if raw.empty:
        raise RuntimeError("Failed to download SPY data.")
    if hasattr(raw.index, "tz") and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)

    close = raw["Close"]
    spy = pd.DataFrame(index=raw.index)
    spy["spy_return_1d"] = close.pct_change(1)
    spy["spy_return_5d"] = close.pct_change(5)
    spy["spy_ma_ratio"]  = close / close.rolling(20).mean()
    spy = spy.dropna()
    spy.to_csv(SPY_CACHE)
    print(f"         SPY features cached -> {SPY_CACHE}")
    return spy


# ── Step 1: Load master dataset ───────────────────────────────────────────────

def load_data(tickers: list[str], csv_path: str,
              start_date: str | None = None) -> dict[str, pd.DataFrame]:
    print(f"\n[Step 1] Loading {csv_path} ...")
    raw = pd.read_csv(csv_path, parse_dates=["date"])
    available = raw["ticker"].unique().tolist()

    missing = [t for t in tickers if t not in available]
    if missing:
        raise ValueError(f"Tickers not found in {csv_path}: {missing}\n"
                         f"Available: {sorted(available)}")

    if start_date:
        raw = raw[raw["date"] >= pd.Timestamp(start_date)]
        print(f"         Filtered to {start_date} onwards")

    out = {}
    for ticker in tickers:
        df = raw[raw["ticker"] == ticker].copy()
        df = df.sort_values("date").set_index("date")
        df.index.name = "Date"
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        out[ticker] = df

    total_rows = sum(len(v) for v in out.values())
    dates = next(iter(out.values()))
    print(f"         Tickers : {tickers}")
    print(f"         Rows    : {total_rows} ({len(tickers)} tickers × "
          f"{len(dates)} days each)")
    print(f"         Range   : {dates.index[0].date()} -> {dates.index[-1].date()}")
    return out


# ── Step 2: Feature engineering ───────────────────────────────────────────────

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
    stoch_k = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def _bb(close, period=20, n=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + n * std
    lower = mid - n * std
    bb_width = (2 * n * std) / mid.replace(0, np.nan)
    bb_pct   = (close - lower) / (upper - lower).replace(0, np.nan)
    return bb_width, bb_pct

def _atr_pct(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / close


def _build_features_single(df: pd.DataFrame, spy: pd.DataFrame,
                            use_sentiment: bool) -> pd.DataFrame:
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    out = pd.DataFrame(index=df.index)

    # Returns
    out["return_1d"]  = close.pct_change(1)
    out["return_2d"]  = close.pct_change(2)
    out["return_5d"]  = close.pct_change(5)

    # Moving averages
    ma5  = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    out["price_to_ma5"]  = close / ma5.replace(0, np.nan)
    out["price_to_ma20"] = close / ma20.replace(0, np.nan)
    out["ma5_to_ma20"]   = ma5 / ma20.replace(0, np.nan)

    # Momentum
    out["rsi"]       = _rsi(close)
    out["macd_diff"] = _macd_diff(close)
    out["stoch_k"], out["stoch_d"] = _stoch(high, low, close)

    # Volatility
    out["bb_width"], out["bb_pct"] = _bb(close)
    out["atr_pct"] = _atr_pct(high, low, close)

    # Volume
    out["volume_ratio"] = volume / volume.rolling(20).mean()

    # Lags
    out["return_lag1"] = out["return_1d"].shift(1)
    out["return_lag2"] = out["return_1d"].shift(2)
    out["return_lag3"] = out["return_1d"].shift(3)
    out["rsi_lag1"]    = out["rsi"].shift(1)
    out["rsi_lag2"]    = out["rsi"].shift(2)
    out["rsi_lag3"]    = out["rsi"].shift(3)

    # Sentiment — carry real scores where available, zero elsewhere
    sent_cols = [c for c in SENTIMENT_FEATURES if c != "has_sentiment"]
    for col in sent_cols:
        out[col] = df[col].values if (use_sentiment and col in df.columns) else 0.0
    # has_sentiment = 1 on rows where FinBERT actually ran, so XGBoost can condition on it
    if use_sentiment and "finbert_score_mean" in df.columns:
        out["has_sentiment"] = (df["finbert_score_mean"].values != 0).astype(float)
    else:
        out["has_sentiment"] = 0.0

    # SPY market features
    for col in MARKET_FEATURES:
        out[col] = spy[col].reindex(out.index, fill_value=0.0)

    # Label
    out["target"] = (close.shift(-1) > close).astype(int)
    return out


def build_features(ticker_data: dict[str, pd.DataFrame], use_sentiment: bool) -> pd.DataFrame:
    print(f"\n[Step 2] Engineering features for {len(ticker_data)} ticker(s) ...")
    spy = load_spy_features()

    frames = []
    for ticker, df in ticker_data.items():
        feat = _build_features_single(df, spy, use_sentiment)
        feat["ticker"] = ticker
        frames.append(feat)

    combined = pd.concat(frames).sort_index()

    sentiment_status = "ON" if use_sentiment else "OFF (zeroed)"
    print(f"         {len(TECHNICAL_FEATURES)} technical + "
          f"{len(SENTIMENT_FEATURES)} sentiment + "
          f"{len(MARKET_FEATURES)} market (SPY) features "
          f"(sentiment {sentiment_status})")
    print(f"         {len(combined)} total rows across all tickers")

    if use_sentiment:
        nonzero = int((combined["has_sentiment"] == 1).sum())
        total   = len(combined)
        print(f"         Sentiment coverage: {nonzero}/{total} rows "
              f"({nonzero/total*100:.1f}%) have real FinBERT scores")
    return combined


# ── Step 3: Train / val / test split ─────────────────────────────────────────

def time_series_split(df: pd.DataFrame, val_ratio=0.15, test_ratio=0.15):
    print(f"\n[Step 3] Splitting data (time-series by date, no shuffle) ...")
    clean = df[ALL_FEATURES + ["target"]].dropna()

    unique_dates    = sorted(clean.index.unique())
    n               = len(unique_dates)
    val_start_date  = unique_dates[int(n * (1 - test_ratio - val_ratio))]
    test_start_date = unique_dates[int(n * (1 - test_ratio))]

    train = clean[clean.index <  val_start_date]
    val   = clean[(clean.index >= val_start_date) & (clean.index < test_start_date)]
    test  = clean[clean.index >= test_start_date]

    print(f"         Train : {len(train):>5} rows  (up to {val_start_date.date()})")
    print(f"         Val   : {len(val):>5} rows  "
          f"({val_start_date.date()} -> {test_start_date.date()})")
    print(f"         Test  : {len(test):>5} rows  "
          f"({test_start_date.date()} -> {unique_dates[-1].date()})")
    print(f"         Label balance — train: {train['target'].mean():.1%} bullish  |  "
          f"test: {test['target'].mean():.1%} bullish")
    return train, val, test


# ── Step 4: Train XGBoost ─────────────────────────────────────────────────────

def train_model(train: pd.DataFrame, val: pd.DataFrame) -> xgb.XGBClassifier:
    print(f"\n[Step 4] Training XGBoost ...")
    X_train, y_train = train[ALL_FEATURES], train["target"]
    X_val,   y_val   = val[ALL_FEATURES],   val["target"]

    model = xgb.XGBClassifier(
        n_estimators          = 1000,
        learning_rate         = 0.01,   # slower learning = less overfit
        max_depth             = 3,      # shallower trees = less overfit
        subsample             = 0.7,
        colsample_bytree      = 0.7,
        min_child_weight      = 10,     # require more samples per leaf
        gamma                 = 0.2,    # higher split threshold
        reg_alpha             = 0.5,    # stronger L1
        reg_lambda            = 2.0,    # stronger L2
        eval_metric           = "auc",
        early_stopping_rounds = 100,    # more patience
        random_state          = 42,
        verbosity             = 0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print(f"         Best iteration : {model.best_iteration}")
    print(f"         Best val AUC   : {model.best_score:.4f}")
    return model


# ── Step 5: Evaluate ──────────────────────────────────────────────────────────

def evaluate(model: xgb.XGBClassifier, test: pd.DataFrame) -> float:
    print(f"\n[Step 5] Evaluating on test set ...")
    X_test, y_test = test[ALL_FEATURES], test["target"]
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    print(f"\n  --- CONFUSION MATRIX ---")
    print(f"                    Predicted 0   Predicted 1")
    print(f"  Actual 0 (bear)     {TN:>6}        {FP:>6}")
    print(f"  Actual 1 (bull)     {FN:>6}        {TP:>6}")

    auc = roc_auc_score(y_test, y_proba)
    print(f"\n  --- METRICS ---")
    print(f"  Accuracy   : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  F1 (bull)  : {f1_score(y_test, y_pred):.4f}")
    print(f"  Precision  : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall     : {recall_score(y_test, y_pred):.4f}")

    print(f"\n  --- TOP 10 FEATURE IMPORTANCES ---")
    imp = sorted(zip(model.feature_names_in_, model.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(imp[:10], 1):
        bar  = "#" * int(score * 300)
        tag  = "(sentiment)" if name in SENTIMENT_FEATURES else \
               "(market)"    if name in MARKET_FEATURES    else ""
        print(f"  {i:>2}. {name:<28} {score:.4f}  {bar} {tag}")

    # Feature importance plot
    feat_names  = [x[0] for x in imp]
    feat_scores = [x[1] for x in imp]
    colors = ["#e07b54" if n in SENTIMENT_FEATURES else
              "#2ca02c" if n in MARKET_FEATURES    else
              "#4c72b0" for n in feat_names]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feat_names[::-1], feat_scores[::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance (weight)")
    ax.set_title("XGBoost Feature Importance\n"
                 "(blue = technical, orange = sentiment, green = market/SPY)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Feature importance plot saved -> feature_importance.png")
    return auc


# ── Step 6: Save model ────────────────────────────────────────────────────────

def save_model(model: xgb.XGBClassifier, label: str, auc: float) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename  = f"xgb_{label.lower()}_auc{auc:.4f}_{timestamp}.json"
    model.save_model(filename)
    print(f"\n[Step 6] Model saved -> {filename}")
    return filename


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args          = sys.argv[1:]
    use_sentiment = "--no-sentiment" not in args
    csv_path      = MASTER_CSV
    start_date    = "2025-06-01"   # default: trim to window where FinBERT scores exist

    if "--csv" in args:
        idx = args.index("--csv")
        if idx + 1 < len(args):
            csv_path = args[idx + 1]
    if "--start-date" in args:
        idx = args.index("--start-date")
        if idx + 1 < len(args):
            start_date = args[idx + 1]

    # Collect ticker args (anything not starting with -- and not a flag value)
    flag_values = set()
    for flag in ["--csv", "--start-date"]:
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                flag_values.add(args[idx + 1])
    raw_args = [a for a in args if not a.startswith("--") and a not in flag_values]

    if not raw_args or raw_args[0].upper() == "ALL":
        tickers = ALL_TICKERS
        label   = "all"
    else:
        tickers = [t.upper() for t in raw_args]
        label   = "_".join(tickers).lower()

    print("=" * 60)
    print(f"  XGBoost Stock Direction Classifier")
    print(f"  Tickers  : {tickers}")
    print(f"  Dataset  : {csv_path}")
    print(f"  Sentiment: {'ON' if use_sentiment else 'OFF'}")
    if start_date:
        print(f"  From     : {start_date}")
    print("=" * 60)

    ticker_data      = load_data(tickers, csv_path, start_date=start_date)
    feature_df       = build_features(ticker_data, use_sentiment)
    train, val, test = time_series_split(feature_df)
    model            = train_model(train, val)
    auc              = evaluate(model, test)
    save_model(model, label, auc)

    print(f"\n{'=' * 60}")
    print(f"  Training complete.  Test AUC = {auc:.4f}")
    old_auc = 0.6920
    diff    = auc - old_auc
    print(f"  vs old model AUC  = {old_auc:.4f}  ({'+' if diff>=0 else ''}{diff:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()

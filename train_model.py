"""
XGBoost Stock Direction Classifier
====================================
Loads OHLCV + pre-scored FinBERT sentiment from master_dataset_with_sentiment.csv,
engineers technical indicators, and trains an XGBoost binary classifier to predict
next-day price direction (up=1 / down=0).

Usage
-----
    python train_model.py                                          # AAPL only
    python train_model.py TSLA                                     # single ticker
    python train_model.py ALL                                      # all 10 tickers combined
    python train_model.py AAPL TSLA NVDA                           # specific tickers combined
    python train_model.py ALL --no-sentiment                       # technical + SPY only
    python train_model.py ALL --start-date 2025-06-01             # trim to sentiment window
    python train_model.py ALL --csv other_master.csv               # custom input file
    python train_model.py ALL --tune                               # Optuna hyperparameter search
    python train_model.py ALL --walk-forward                       # walk-forward cross-validation
    python train_model.py ALL --finetune                           # fine-tune latest saved model
    python train_model.py ALL --finetune --finetune-model xgb.json # fine-tune specific model
    python train_model.py ALL --finetune --finetune-months 3       # fine-tune on last 3 months
"""

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score,
    f1_score, precision_score, recall_score,
)
import matplotlib.pyplot as plt
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

warnings.filterwarnings("ignore", category=UserWarning)

MASTER_CSV = "master_dataset_with_sentiment.csv"

ALL_TICKERS = ["AAPL", "AMD", "AMZN", "AVGO", "GOOGL",
               "META", "MSFT", "NVDA", "ORCL", "TSLA"]

# Full feature set matching xgb_stock_model baseline + extra
TECHNICAL_FEATURES = [
    "return_1d", "return_5d",            # dropped: return_2d (redundant)
    "price_to_ma5", "ma5_to_ma20",       # dropped: price_to_ma20 (correlated)
    "rsi", "macd_diff", "stoch_k",       # dropped: stoch_d (derived from stoch_k)
    "bb_pct", "atr_pct", "volume_ratio", # dropped: bb_width (correlated with bb_pct)
    "return_lag1", "return_lag2", "return_lag3",
    "rsi_lag1",                          # dropped: rsi_lag2, rsi_lag3 (diminishing signal)
]

SENTIMENT_FEATURES = [
    "finbert_score_mean", "finbert_pos_mean", "finbert_neg_mean",
    "news_count", "sentiment_lag1", "sentiment_lag2",
]

MARKET_FEATURES = [
    "spy_return_1d",
    "spy_return_5d",
    "spy_ma_ratio",
    "vix_change",
    "vix_ma_ratio",
]

ALL_FEATURES = TECHNICAL_FEATURES + SENTIMENT_FEATURES + MARKET_FEATURES

SPY_CACHE = Path(".cache/spy_features.csv")
VIX_CACHE = Path(".cache/vix_features.csv")


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


# ── VIX market features ───────────────────────────────────────────────────────

def load_vix_features() -> pd.DataFrame:
    if VIX_CACHE.exists():
        print(f"         VIX features loaded from cache ({VIX_CACHE})")
        return pd.read_csv(VIX_CACHE, index_col=0, parse_dates=True)

    print(f"         Downloading VIX data from yfinance ...")
    VIX_CACHE.parent.mkdir(exist_ok=True)
    raw = yf.Ticker("^VIX").history(period="10y", interval="1d", auto_adjust=True)
    if raw.empty:
        raise RuntimeError("Failed to download VIX data.")
    if hasattr(raw.index, "tz") and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)

    close = raw["Close"]
    vix = pd.DataFrame(index=raw.index)
    vix["vix_change"]   = close.pct_change(1)
    vix["vix_ma_ratio"] = close / close.rolling(20).mean()  # relative to recent history
    vix = vix.dropna()
    vix.to_csv(VIX_CACHE)
    print(f"         VIX features cached -> {VIX_CACHE}")
    return vix


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
    print(
        f"         Range   : {dates.index[0].date()} -> {dates.index[-1].date()}")
    return out


# ── Step 2: Feature engineering ───────────────────────────────────────────────

def _rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/period,
                            min_periods=period, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/period,
                               min_periods=period, adjust=False).mean()
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
    bb_pct = (close - lower) / (upper - lower).replace(0, np.nan)
    return bb_width, bb_pct


def _atr_pct(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(),
                   (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / close


def _build_features_single(df: pd.DataFrame, spy: pd.DataFrame,
                           vix: pd.DataFrame, use_sentiment: bool) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    out = pd.DataFrame(index=df.index)

    # Returns
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)

    # Moving averages
    ma5  = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    out["price_to_ma5"] = close / ma5.replace(0, np.nan)
    out["ma5_to_ma20"]  = ma5 / ma20.replace(0, np.nan)

    # Momentum
    out["rsi"]      = _rsi(close)
    out["macd_diff"] = _macd_diff(close)
    out["stoch_k"], _ = _stoch(high, low, close)

    # Volatility
    _, out["bb_pct"] = _bb(close)
    out["atr_pct"]   = _atr_pct(high, low, close)

    # Volume
    out["volume_ratio"] = volume / volume.rolling(20).mean()

    # Lags
    out["return_lag1"] = out["return_1d"].shift(1)
    out["return_lag2"] = out["return_1d"].shift(2)
    out["return_lag3"] = out["return_1d"].shift(3)
    out["rsi_lag1"]    = out["rsi"].shift(1)

    # Sentiment
    for col in SENTIMENT_FEATURES:
        out[col] = df[col].values if (use_sentiment and col in df.columns) else 0.0

    # SPY + VIX market features
    market = spy.join(vix, how="outer")
    for col in MARKET_FEATURES:
        out[col] = market[col].reindex(out.index, fill_value=0.0)

    # Label
    out["target"] = (close.shift(-1) > close).astype(int)
    return out


def build_features(ticker_data: dict[str, pd.DataFrame], use_sentiment: bool) -> pd.DataFrame:
    print(f"\n[Step 2] Engineering features for {len(ticker_data)} ticker(s) ...")
    spy = load_spy_features()
    vix = load_vix_features()

    frames = []
    for ticker, df in ticker_data.items():
        feat = _build_features_single(df, spy, vix, use_sentiment)
        feat["ticker"] = ticker
        frames.append(feat)

    combined = pd.concat(frames).sort_index()

    sentiment_status = "ON" if use_sentiment else "OFF (zeroed)"
    print(f"         {len(TECHNICAL_FEATURES)} technical + "
          f"{len(SENTIMENT_FEATURES)} sentiment + "
          f"{len(MARKET_FEATURES)} market (SPY+VIX) features "
          f"(sentiment {sentiment_status})")
    print(f"         {len(combined)} total rows across all tickers")

    if use_sentiment:
        nonzero = int((combined["finbert_score_mean"] != 0).sum())
        total = len(combined)
        print(f"         Sentiment coverage: {nonzero}/{total} rows "
              f"({nonzero/total*100:.1f}%) have real FinBERT scores")
    return combined


# ── Step 3: Train / val / test split ─────────────────────────────────────────

def time_series_split(df: pd.DataFrame, val_ratio=0.15, test_ratio=0.15):
    print(f"\n[Step 3] Splitting data (time-series by date, no shuffle) ...")
    clean = df[ALL_FEATURES + ["target"]].dropna()

    unique_dates = sorted(clean.index.unique())
    n = len(unique_dates)
    val_start_date = unique_dates[int(n * (1 - test_ratio - val_ratio))]
    test_start_date = unique_dates[int(n * (1 - test_ratio))]

    train = clean[clean.index < val_start_date]
    val = clean[(clean.index >= val_start_date) &
                (clean.index < test_start_date)]
    test = clean[clean.index >= test_start_date]

    print(
        f"         Train : {len(train):>5} rows  (up to {val_start_date.date()})")
    print(f"         Val   : {len(val):>5} rows  "
          f"({val_start_date.date()} -> {test_start_date.date()})")
    print(f"         Test  : {len(test):>5} rows  "
          f"({test_start_date.date()} -> {unique_dates[-1].date()})")
    print(f"         Label balance — train: {train['target'].mean():.1%} bullish  |  "
          f"test: {test['target'].mean():.1%} bullish")
    return train, val, test


# ── Step 3b: Walk-forward validation ─────────────────────────────────────────

def walk_forward_eval(feature_df: pd.DataFrame, params: dict,
                      n_splits: int = 5) -> float:
    """
    Train on rolling windows and average AUC across n_splits folds.
    Gives a more honest performance estimate than a single test period.
    """
    from sklearn.model_selection import TimeSeriesSplit
    print(f"\n[Walk-Forward] {n_splits}-fold time-series cross-validation ...")
    clean = feature_df[ALL_FEATURES + ["target"]].dropna()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(clean)):
        tr_full = clean.iloc[train_idx]
        test = clean.iloc[test_idx]

        # Internal val split for early stopping (last 15% of train)
        cut = int(len(tr_full) * 0.85)
        tr, vl = tr_full.iloc[:cut], tr_full.iloc[cut:]

        m = xgb.XGBClassifier(**params, eval_metric="auc",
                              early_stopping_rounds=50,
                              random_state=42, verbosity=0)
        m.fit(tr[ALL_FEATURES], tr["target"],
              eval_set=[(vl[ALL_FEATURES], vl["target"])], verbose=False)

        auc = roc_auc_score(test["target"],
                            m.predict_proba(test[ALL_FEATURES])[:, 1])
        aucs.append(auc)
        print(f"  Fold {fold+1}/{n_splits}  "
              f"{test.index[0].date()} -> {test.index[-1].date()}  "
              f"AUC = {auc:.4f}  ({len(tr)} train rows)")

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    print(f"\n  Walk-forward AUC : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  (more reliable than single test split)")
    return mean_auc


# ── Step 3c: Optuna hyperparameter tuning ────────────────────────────────────

def optimize_hyperparams(feature_df: pd.DataFrame,
                         n_trials: int = 50) -> dict:
    """
    Use Optuna + TimeSeriesSplit to find the best XGBoost hyperparameters.
    Returns the best param dict to pass into train_model().
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed. Run: pip install optuna")
        return {}

    from sklearn.model_selection import TimeSeriesSplit
    clean = feature_df[ALL_FEATURES + ["target"]].dropna()
    tscv = TimeSeriesSplit(n_splits=5)

    print(f"\n[Optuna] Searching hyperparameters ({n_trials} trials) ...")

    def objective(trial):
        params = {
            "n_estimators":       300,
            "learning_rate":      trial.suggest_float("learning_rate",    0.005, 0.15, log=True),
            "max_depth":          trial.suggest_int("max_depth",           2, 6),
            "subsample":          trial.suggest_float("subsample",         0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree",  0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight",    3, 30),
            "gamma":              trial.suggest_float("gamma",             0.0, 0.5),
            "reg_alpha":          trial.suggest_float("reg_alpha",         0.0, 3.0),
            "reg_lambda":         trial.suggest_float("reg_lambda",        0.5, 4.0),
            "eval_metric":        "auc",
            "early_stopping_rounds": 30,
            "random_state":       42,
            "verbosity":          0,
        }
        fold_aucs = []
        for train_idx, val_idx in tscv.split(clean):
            tr, vl = clean.iloc[train_idx], clean.iloc[val_idx]
            cut = int(len(tr) * 0.85)
            tr_tr, tr_vl = tr.iloc[:cut], tr.iloc[cut:]
            m = xgb.XGBClassifier(**params)
            m.fit(tr_tr[ALL_FEATURES], tr_tr["target"],
                  eval_set=[(tr_vl[ALL_FEATURES], tr_vl["target"])],
                  verbose=False)
            fold_aucs.append(roc_auc_score(
                vl["target"], m.predict_proba(vl[ALL_FEATURES])[:, 1]))
        return float(np.mean(fold_aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n  Best cross-val AUC : {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in best.items():
        print(f"    {k:<22} = {v}")
    return best


# ── Step 4: Train XGBoost ─────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "n_estimators":          1000,
    "learning_rate":         0.01,
    "max_depth":             3,
    "subsample":             0.7,
    "colsample_bytree":      0.7,
    "min_child_weight":      10,
    "gamma":                 0.2,
    "reg_alpha":             0.5,
    "reg_lambda":            2.0,
    "scale_pos_weight":      1.0, # 1.0 = neutral; increase to penalise missing bears
    "eval_metric":           "auc",
    "early_stopping_rounds": 100,
    "random_state":          42,
    "verbosity":             0,
}


def train_model(train: pd.DataFrame, val: pd.DataFrame,
                params: dict | None = None) -> xgb.XGBClassifier:
    print(f"\n[Step 4] Training XGBoost ...")
    X_train, y_train = train[ALL_FEATURES], train["target"]
    X_val,   y_val = val[ALL_FEATURES],   val["target"]

    p = {**DEFAULT_PARAMS, **(params or {})}
    model = xgb.XGBClassifier(**p)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print(f"         Best iteration : {model.best_iteration}")
    print(f"         Best val AUC   : {model.best_score:.4f}")
    return model


# ── Step 4b: Fine-tune existing model ────────────────────────────────────────

def _optuna_finetune(X_ft, y_ft, X_val, y_val,
                     base_model_path: str, n_trials: int = 30) -> dict:
    """Use Optuna to find best hyperparameters for the fine-tuning step."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed — using defaults. Run: pip install optuna")
        return {}

    print(f"  [Optuna] Searching fine-tune hyperparameters ({n_trials} trials) ...")

    def objective(trial):
        params = {
            "n_estimators":          200,
            "learning_rate":         trial.suggest_float("learning_rate",  0.001, 0.02, log=True),
            "max_depth":             trial.suggest_int("max_depth",         2, 5),
            "subsample":             trial.suggest_float("subsample",       0.5, 1.0),
            "colsample_bytree":      trial.suggest_float("colsample_bytree",0.5, 1.0),
            "min_child_weight":      trial.suggest_int("min_child_weight",  3, 20),
            "gamma":                 trial.suggest_float("gamma",           0.0, 0.5),
            "reg_alpha":             trial.suggest_float("reg_alpha",       0.0, 3.0),
            "reg_lambda":            trial.suggest_float("reg_lambda",      0.5, 4.0),
            "eval_metric":           "auc",
            "early_stopping_rounds": 20,
            "random_state":          42,
            "verbosity":             0,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_ft, y_ft,
              eval_set=[(X_val, y_val)],
              xgb_model=base_model_path,
              verbose=False)
        return m.best_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"  [Optuna] Best fine-tune AUC : {study.best_value:.4f}")
    for k, v in best.items():
        print(f"    {k:<22} = {v}")
    return best


def finetune_model(base_model_path: str, train: pd.DataFrame, val: pd.DataFrame,
                   months: int = 6, optimize: bool = False) -> xgb.XGBClassifier:
    """
    Continue training from a saved model on the most recent N months of data.
    Pass optimize=True to run Optuna on the recent window before fine-tuning.
    """
    import glob, os
    if base_model_path is None:
        candidates = [f for f in glob.glob("xgb_*.json")
                      if f != "xgb_stock_model.json"]
        if not candidates:
            raise FileNotFoundError("No saved model found. Train a model first.")
        base_model_path = max(candidates, key=os.path.getmtime)

    print(f"\n[Step 4b] Fine-tuning: {base_model_path}")

    cutoff       = train.index.max() - pd.DateOffset(months=months)
    recent       = train[train.index >= cutoff]
    X_ft, y_ft   = recent[ALL_FEATURES], recent["target"]
    X_val, y_val = val[ALL_FEATURES],    val["target"]

    print(f"         Recent window : last {months} months  "
          f"({recent.index.min().date()} -> {recent.index.max().date()})")
    print(f"         Fine-tune rows: {len(recent)}")

    # Optuna search on the recent window if requested
    best_params = {}
    if optimize:
        best_params = _optuna_finetune(X_ft, y_ft, X_val, y_val,
                                       base_model_path, n_trials=30)

    neg, pos = int((y_ft == 0).sum()), int((y_ft == 1).sum())
    ft_params = {
        "n_estimators":          200,
        "learning_rate":         0.005,
        "max_depth":             3,
        "subsample":             0.7,
        "colsample_bytree":      0.7,
        "min_child_weight":      5,
        "gamma":                 0.2,
        "reg_alpha":             0.5,
        "reg_lambda":            2.0,
        "scale_pos_weight":      neg / pos if pos > 0 else 1.0,
        "eval_metric":           "auc",
        "early_stopping_rounds": 30,
        "random_state":          42,
        "verbosity":             0,
        **best_params,           # Optuna overrides defaults if run
    }
    print(f"         scale_pos_weight = {ft_params['scale_pos_weight']:.3f}  "
          f"(bear={neg}, bull={pos})")

    model = xgb.XGBClassifier(**ft_params)
    model.fit(X_ft, y_ft,
              eval_set=[(X_val, y_val)],
              xgb_model=base_model_path,
              verbose=20)

    print(f"         Best iteration : {model.best_iteration}")
    print(f"         Best val AUC   : {model.best_score:.4f}")
    return model


# ── Step 5: Evaluate ──────────────────────────────────────────────────────────

def _best_threshold(y_true, y_proba) -> float:
    """Find threshold that maximises balanced accuracy, searching 0.45–0.65."""
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.45, 0.66, 0.01):
        pred        = (y_proba >= t).astype(int)
        n_bull_pred = pred.sum()
        n_bear_pred = len(pred) - n_bull_pred
        if n_bull_pred == 0 or n_bear_pred == 0:
            continue   # skip degenerate all-one-class solutions
        bear_recall = (pred[y_true == 0] == 0).mean()
        bull_recall = (pred[y_true == 1] == 1).mean()
        bal = (bear_recall + bull_recall) / 2
        if bal > best_score:
            best_score, best_t = bal, t
    return round(best_t, 2)


def evaluate(model: xgb.XGBClassifier, test: pd.DataFrame,
             val: pd.DataFrame | None = None) -> float:
    print(f"\n[Step 5] Evaluating on test set ...")
    X_test, y_test = test[ALL_FEATURES], test["target"]
    y_proba = model.predict_proba(X_test)[:, 1]

    # Find best threshold on val set (not test) to avoid leakage
    if val is not None:
        X_val, y_val = val[ALL_FEATURES], val["target"]
        val_proba    = model.predict_proba(X_val)[:, 1]
        threshold    = _best_threshold(y_val.values, val_proba)
        print(f"  Threshold optimised on val set: {threshold:.2f}  (default=0.50)")
    else:
        threshold = 0.50

    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    print(f"\n  --- CONFUSION MATRIX (threshold={threshold:.2f}) ---")
    print(f"                    Predicted 0   Predicted 1")
    print(f"  Actual 0 (bear)     {TN:>6}        {FP:>6}")
    print(f"  Actual 1 (bull)     {FN:>6}        {TP:>6}")
    print(f"  Bear recall : {TN/(TN+FP):.1%}  |  Bull recall : {TP/(TP+FN):.1%}")
    print(f"  Predicted bull: {y_pred.mean():.1%}  |  Actual bull: {y_test.mean():.1%}")

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
        bar = "#" * int(score * 300)
        tag = "(sentiment)" if name in SENTIMENT_FEATURES else \
            "(market)" if name in MARKET_FEATURES else ""
        print(f"  {i:>2}. {name:<28} {score:.4f}  {bar} {tag}")

    # Feature importance plot
    feat_names = [x[0] for x in imp]
    feat_scores = [x[1] for x in imp]
    colors = ["#e07b54" if n in SENTIMENT_FEATURES else
              "#2ca02c" if n in MARKET_FEATURES else
              "#4c72b0" for n in feat_names]
    _, ax = plt.subplots(figsize=(10, 8))
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
    filename = f"xgb_{label.lower()}_auc{auc:.4f}_{timestamp}.json"
    model.save_model(filename)
    print(f"\n[Step 6] Model saved -> {filename}")
    return filename


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    use_sentiment   = "--no-sentiment" not in args
    run_tune        = "--tune" in args
    run_wf          = "--walk-forward" in args
    run_finetune    = "--finetune" in args
    csv_path        = MASTER_CSV
    start_date      = None
    finetune_model_path  = None
    finetune_months = 6

    if "--csv" in args:
        idx = args.index("--csv")
        if idx + 1 < len(args):
            csv_path = args[idx + 1]
    if "--start-date" in args:
        idx = args.index("--start-date")
        if idx + 1 < len(args):
            start_date = args[idx + 1]
    if "--finetune-model" in args:
        idx = args.index("--finetune-model")
        if idx + 1 < len(args):
            finetune_model_path = args[idx + 1]
    if "--finetune-months" in args:
        idx = args.index("--finetune-months")
        if idx + 1 < len(args):
            finetune_months = int(args[idx + 1])

    # Collect ticker args (anything not starting with -- and not a flag value)
    flag_values = set()
    for flag in ["--csv", "--start-date", "--finetune-model", "--finetune-months"]:
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                flag_values.add(args[idx + 1])
    raw_args = [a for a in args if not a.startswith(
        "--") and a not in flag_values]

    if not raw_args or raw_args[0].upper() == "ALL":
        tickers = ALL_TICKERS
        label = "all"
    else:
        tickers = [t.upper() for t in raw_args]
        label = "_".join(tickers).lower()

    print("=" * 60)
    print(f"  XGBoost Stock Direction Classifier")
    print(f"  Tickers  : {tickers}")
    print(f"  Dataset  : {csv_path}")
    print(f"  Sentiment: {'ON' if use_sentiment else 'OFF'}")
    if start_date:
        print(f"  From     : {start_date}")
    print("=" * 60)

    ticker_data = load_data(tickers, csv_path, start_date=start_date)
    feature_df = build_features(ticker_data, use_sentiment)

    # Optuna tuning — find best hyperparameters via cross-validation
    best_params = {}
    if run_tune:
        best_params = optimize_hyperparams(feature_df, n_trials=50)

    # Walk-forward evaluation — honest multi-period AUC
    if run_wf:
        wf_params = {k: v for k, v in {**DEFAULT_PARAMS, **best_params}.items()
                     if k not in ("eval_metric", "early_stopping_rounds",
                                  "random_state", "verbosity")}
        walk_forward_eval(feature_df, params=wf_params)

    # Final model — train on full train+val, evaluate on test
    train, val, test = time_series_split(feature_df)

    if run_finetune:
        model = finetune_model(finetune_model_path, train, val,
                               months=finetune_months, optimize=run_tune)
    else:
        model = train_model(train, val, params=best_params or None)

    auc = evaluate(model, test, val=val)
    save_model(model, label, auc)

    print(f"\n{'=' * 60}")
    print(f"  Training complete.  Test AUC = {auc:.4f}")
    old_auc = 0.6920
    diff = auc - old_auc
    print(
        f"  vs old model AUC  = {old_auc:.4f}  ({'+' if diff>=0 else ''}{diff:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()

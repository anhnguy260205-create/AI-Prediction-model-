"""
Daily Update Pipeline
=====================
Runs every morning to collect fresh news, score sentiment,
update the price dataset, and print next-day predictions.

Steps
-----
    1. Collect new headlines (Finnhub + Alpaca, skips cached months)
    2. Clean headlines
    3. Score new headlines with FinBERT (skips already-scored ones)
    4. Download latest OHLCV prices
    5. Run predictions with the best saved model

Usage
-----
    python daily_update.py              # full pipeline
    python daily_update.py --no-retrain # skip model retraining
"""

import sys
import subprocess
import os
from datetime import datetime
from pathlib import Path

PYTHON      = sys.executable
PROJECT_DIR = Path(__file__).parent
LOG_DIR     = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

TODAY       = datetime.now().strftime("%Y-%m-%d")
LOG_FILE    = LOG_DIR / f"daily_{datetime.now().strftime('%Y%m%d_%H%M')}.log"


def log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(script: str, *args) -> bool:
    cmd = [PYTHON, str(PROJECT_DIR / script)] + list(args)
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_DIR,
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace")
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            log(f"  {line}")
    if result.returncode != 0:
        log(f"  ERROR (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-10:]:
                log(f"  {line}")
        return False
    return True


def update_collect_news_end_date():
    """Patch END_DATE in collect_news.py to today."""
    path = PROJECT_DIR / "collect_news.py"
    text = path.read_text(encoding="utf-8")
    import re
    updated = re.sub(
        r'END_DATE\s*=\s*"[\d-]+"',
        f'END_DATE    = "{TODAY}"',
        text
    )
    path.write_text(updated, encoding="utf-8")
    log(f"  collect_news.py END_DATE -> {TODAY}")


def find_best_model() -> str:
    import glob, re
    models = [m for m in glob.glob(str(PROJECT_DIR / "xgb_*.json"))
              if "xgb_stock_model.json" not in m]
    if not models:
        return ""
    def _auc(path):
        m = re.search(r"auc([\d.]+)", path)
        return float(m.group(1)) if m else 0.0
    return max(models, key=_auc)


def main():
    log("=" * 55)
    log(f"  Daily Update  —  {TODAY}")
    log("=" * 55)

    # ── Step 1: Update end date and collect news ──────────────
    log("\n[Step 1] Collecting latest news ...")
    update_collect_news_end_date()
    if not run("collect_news.py"):
        log("  WARNING: News collection failed — continuing anyway")

    # ── Step 2: Clean headlines ───────────────────────────────
    log("\n[Step 2] Cleaning headlines ...")
    if not run("clean_news.py"):
        log("  WARNING: Cleaning failed — continuing anyway")

    # ── Step 3: Score with FinBERT ────────────────────────────
    log("\n[Step 3] Scoring new headlines with FinBERT ...")
    if not run("run_finbert.py"):
        log("  WARNING: FinBERT scoring failed — continuing anyway")

    # ── Step 4: Download latest prices ───────────────────────
    log("\n[Step 4] Downloading latest OHLCV prices ...")
    if not run("build_dataset.py"):
        log("  WARNING: Price download failed — predictions may use stale data")

    # ── Step 5: Predict ───────────────────────────────────────
    log("\n[Step 5] Running predictions ...")
    best_model = find_best_model()
    if not best_model:
        log("  ERROR: No saved model found. Run train_model.py first.")
        sys.exit(1)

    log(f"  Using model: {Path(best_model).name}")
    run("predict.py", "ALL", "--model", Path(best_model).name)

    log("\n" + "=" * 55)
    log(f"  Done.  Log saved -> {LOG_FILE.name}")
    log("=" * 55)


if __name__ == "__main__":
    main()

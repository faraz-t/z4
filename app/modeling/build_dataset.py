"""
Steps:
  1. Load the scored CSVs.
  2. For each row, look up the stock price on the discussion date and
     7/14/30 days later (via app.yfinance_utils), and compute forward returns.
  3. Cache price lookups to disk so re-running doesn't re-hit the network
     and so test/train rows that share a (ticker, date) only get looked up once.
  4. Save the merged dataset (with prices/returns attached) to
     data/fake/processed/dataset_{train,test}.csv.

Run before train_model.py, after run_llm.py has produced the scored CSVs.
"""

from __future__ import annotations

import json
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.modeling import yfinance_utils

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent      # app/modeling/
PROJECT_ROOT = APP_DIR.parent.parent           # repo root

PROCESSED_DIR = PROJECT_ROOT / "data" / "real" / "processed"

INPUT_CSV = PROCESSED_DIR / "llm_output.csv"
DATASET_OUT = PROCESSED_DIR / "dataset.csv"

PRICE_CACHE_PATH = PROCESSED_DIR / "price_cache.json"

NO_TICKER_VALUES = {"NONE", "", None}


# ---------------------------------------------------------------------------
# Price cache
# ---------------------------------------------------------------------------
def _load_price_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_price_cache(cache: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def get_returns_cached(
    ticker: str,
    base_date: date,
    cache: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    """Wraps yfinance_utils.compute_returns with a disk-backed cache.

    Returns None (instead of raising) if the price lookup fails.
    """
    key = f"{ticker}|{base_date.isoformat()}"
    if key in cache:
        cached = cache[key]
        if cached is None:
            return None
        return tuple(cached)

    try:
        result = yfinance_utils.compute_returns(ticker, base_date)
        cache[key] = list(result)
        return result
    except Exception as e:
        warnings.warn(f"Price lookup failed for {ticker} on {base_date}: {e}")
        cache[key] = None
        return None


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def attach_returns(df: pd.DataFrame, cache: dict[str, Any]) -> pd.DataFrame:
    """Adds price0/ret_7d/ret_14d/ret_30d columns, dropping rows that can't be priced."""
    df = df.copy()
    df = df[~df["ticker"].isin(NO_TICKER_VALUES)].reset_index(drop=True)

    price0_col, ret7_col, ret14_col, ret30_col = [], [], [], []
    keep_mask = []

    for _, row in df.iterrows():
        base_date = datetime.fromtimestamp(int(row["date"])).date()
        result = get_returns_cached(row["ticker"], base_date, cache)
        if result is None:
            keep_mask.append(False)
            price0_col.append(None)
            ret7_col.append(None)
            ret14_col.append(None)
            ret30_col.append(None)
            continue
        price0, ret7, ret14, ret30 = result
        keep_mask.append(True)
        price0_col.append(price0)
        ret7_col.append(ret7)
        ret14_col.append(ret14)
        ret30_col.append(ret30)

    df["price0"] = price0_col
    df["ret_7d"] = ret7_col
    df["ret_14d"] = ret14_col
    df["ret_30d"] = ret30_col

    dropped = (~pd.Series(keep_mask)).sum()
    if dropped:
        print(f"Dropping {dropped} row(s) with no price data.")

    return df[pd.Series(keep_mask)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cache = _load_price_cache(PRICE_CACHE_PATH)

    print(f"Loading {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV)

    print("Attaching stock returns...")
    df = attach_returns(df, cache)

    _save_price_cache(cache, PRICE_CACHE_PATH)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_OUT, index=False)

    print(f"Saved merged dataset to {DATASET_OUT}")
    print("Next: python -m app.modeling.train_model")


if __name__ == "__main__":
    main()
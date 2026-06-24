"""
Steps:
  1. Load data/fake/processed/dataset_{train,test}.csv (output of build_dataset.py).
  2. Derive the sentiment feature list from features.yaml (every int-typed
     field except `ticker`), and combine with engagement_features from
     model_config.yaml to build X. Targets from model_config.yaml build y.
  3. Train one regressor per return horizon (7d/14d/30d) via MultiOutputRegressor.
  4. Evaluate on the test set and print MAE/R2 per horizon.
  5. Save the trained model to models/reddit_sentiment_model.joblib.

Run after build_dataset.py has produced the merged dataset with forward returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent

LLM_SCHEMA_PATH = APP_DIR.parent / "llm" / "features.yaml"
MODEL_CONFIG_PATH = APP_DIR / "model_config.yaml"

PROCESSED_DIR = PROJECT_ROOT / "data" / "fake" / "processed"
DATASET_TRAIN_CSV = PROCESSED_DIR / "dataset_train.csv"
DATASET_TEST_CSV = PROCESSED_DIR / "dataset_test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_OUT_PATH = MODELS_DIR / "reddit_sentiment_model.joblib"

REQUIRED_MODEL_CONFIG_KEYS = ("engagement_features", "targets")

LLM_SCHEMA_EXCLUDE = {"ticker"}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_sentiment_feature_names(path: Path = LLM_SCHEMA_PATH) -> list[str]:
    """Derives the list of LLM sentiment-score columns from features.yaml.

    features.yaml is a flat YAML list of {name, type, min, max, desc} entries
    (one per LLM-scored field). We take every int-typed entry except `ticker`.
    """
    if not path.exists():
        raise FileNotFoundError(f"LLM schema not found at {path}")

    with open(path, "r") as f:
        schema = yaml.safe_load(f)

    if not isinstance(schema, list):
        raise TypeError(
            f"Expected {path} to parse as a YAML list of field definitions, "
            f"got a {type(schema).__name__}. Has its format changed?"
        )

    names = [
        entry["name"]
        for entry in schema
        if entry.get("type") == "int" and entry.get("name") not in LLM_SCHEMA_EXCLUDE
    ]

    if not names:
        raise ValueError(f"No int-typed fields found in {path} -- check its contents.")

    return names


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"model_config.yaml not found at {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError(
            f"Expected {path} to parse as a YAML mapping (dict) with keys "
            f"{REQUIRED_MODEL_CONFIG_KEYS}, but got a {type(config).__name__}.\n"
            f"Parsed value: {config!r}"
        )

    missing = [k for k in REQUIRED_MODEL_CONFIG_KEYS if k not in config]
    if missing:
        raise KeyError(
            f"{path} is missing required key(s): {missing}. "
            f"Expected top-level keys: {REQUIRED_MODEL_CONFIG_KEYS}"
        )

    return config


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------
def build_xy(
    df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_cols = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Dataset is missing column(s): {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    X = df[feature_cols].astype(float)
    y = df[target_cols].astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> MultiOutputRegressor:
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, random_state=42)
    )
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: MultiOutputRegressor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    target_cols: list[str],
) -> None:
    preds = model.predict(X_test)
    preds_df = pd.DataFrame(preds, columns=target_cols, index=y_test.index)

    print("\nTest-set performance:")
    for col in target_cols:
        mae = mean_absolute_error(y_test[col], preds_df[col])
        r2 = r2_score(y_test[col], preds_df[col]) if len(y_test) > 1 else float("nan")
        print(f"  {col}: MAE={mae:.4f}  R2={r2:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    sentiment_features = load_sentiment_feature_names()
    model_config = load_model_config()
    feature_cols = model_config["engagement_features"] + sentiment_features
    target_cols = model_config["targets"]

    if not DATASET_TRAIN_CSV.exists() or not DATASET_TEST_CSV.exists():
        raise FileNotFoundError(
            f"Merged dataset not found ({DATASET_TRAIN_CSV.name} / {DATASET_TEST_CSV.name}). "
            "Run `python -m app.modeling.build_dataset` first."
        )

    print(f"Loading {DATASET_TRAIN_CSV.name} and {DATASET_TEST_CSV.name}...")
    train_df = pd.read_csv(DATASET_TRAIN_CSV)
    test_df = pd.read_csv(DATASET_TEST_CSV)

    X_train, y_train = build_xy(train_df, feature_cols, target_cols)
    X_test, y_test = build_xy(test_df, feature_cols, target_cols)

    print(f"Features: {feature_cols}")
    print(f"Training on {len(X_train)} rows, evaluating on {len(X_test)} rows...")
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test, target_cols)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": feature_cols, "targets": target_cols},
        MODEL_OUT_PATH,
    )
    print(f"\nSaved trained model to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
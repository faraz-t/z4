"""
Steps:
  1. Load data/real/processed/dataset.csv (output of build_dataset.py).
  2. Derive the sentiment feature list from features.yaml (every int-typed
     field except `ticker`), and combine with engagement_features from
     model_config.yaml to build X. Targets from model_config.yaml build y.
  3. Split the dataset chronologically (80% train, 20% test).
  4. Train one regressor per return horizon via MultiOutputRegressor.
  5. Evaluate on the test set and print MAE/R2 per horizon.
  6. Save the trained model to models/reddit_sentiment_model.joblib.
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

PROCESSED_DIR = PROJECT_ROOT / "data" / "real" / "processed"
DATASET_CSV = PROCESSED_DIR / "dataset.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_OUT_PATH = MODELS_DIR / "reddit_sentiment_model.joblib"

REQUIRED_MODEL_CONFIG_KEYS = ("engagement_features", "targets")

LLM_SCHEMA_EXCLUDE = {"ticker"}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_sentiment_feature_names(path: Path = LLM_SCHEMA_PATH) -> list[str]:
    """Derives the list of LLM sentiment-score columns from features.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"LLM schema not found at {path}")

    with open(path, "r") as f:
        schema = yaml.safe_load(f)

    if not isinstance(schema, list):
        raise TypeError(
            f"Expected {path} to parse as a YAML list, "
            f"got a {type(schema).__name__}."
        )

    names = [
        entry["name"]
        for entry in schema
        if entry.get("type") == "int"
        and entry.get("name") not in LLM_SCHEMA_EXCLUDE
    ]

    if not names:
        raise ValueError(f"No sentiment features found in {path}.")

    return names


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"model_config.yaml not found at {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError(f"Expected {path} to parse as a dictionary.")

    missing = [k for k in REQUIRED_MODEL_CONFIG_KEYS if k not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    return config


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def build_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset missing columns: {missing}")

    X = df[feature_cols].astype(float)
    y = df[target_cols].astype(float)

    return X, y


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> MultiOutputRegressor:
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        )
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
    preds_df = pd.DataFrame(
        preds,
        columns=target_cols,
        index=y_test.index,
    )

    print("\nTest-set performance:")
    for col in target_cols:
        mae = mean_absolute_error(y_test[col], preds_df[col])
        r2 = (
            r2_score(y_test[col], preds_df[col])
            if len(y_test) > 1
            else float("nan")
        )

        print(f"{col:8s} MAE={mae:.4f}   R²={r2:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    sentiment_features = load_sentiment_feature_names()
    model_config = load_model_config()

    feature_cols = (
        model_config["engagement_features"]
        + sentiment_features
    )
    target_cols = model_config["targets"]

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"{DATASET_CSV} not found.\n"
            "Run `python -m app.modeling.build_dataset` first."
        )

    print(f"Loading {DATASET_CSV.name}...")
    df = pd.read_csv(DATASET_CSV)

    # Chronological split (better for financial prediction)
    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = build_xy(train_df, feature_cols, target_cols)
    X_test, y_test = build_xy(test_df, feature_cols, target_cols)

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Training rows:   {len(X_train)}")
    print(f"Testing rows:    {len(X_test)}")

    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test, target_cols)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "targets": target_cols,
        },
        MODEL_OUT_PATH,
    )

    print(f"\nSaved model to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import yaml
from pydantic import Field, create_model
from ollama import Client

MODELS = ["gemma4:e4b", "gemma4:12b"]  # e4b: ~8GB VRAM, 12b: ~16GB VRAM
TYPES = {"int": int, "str": str, "float": float}

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "app" / "reddit" / "processed" / "reddit_comments.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "real" / "processed" / "llm_output.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "app" / "reddit" / "raw"
DEFAULT_SYSTEM_PROMPT = APP_DIR / "system_prompt.md"
DEFAULT_FEATURES = APP_DIR / "features.yaml"


def build_schema(features: list):
    """Build a Pydantic schema from the feature configuration."""
    fields = {}
    for f in features:
        bounds = {k: f[k] for k in ("min", "max") if k in f}
        ge = {"ge": bounds["min"]} if "min" in bounds else {}
        le = {"le": bounds["max"]} if "max" in bounds else {}
        fields[f["name"]] = (TYPES[f["type"]], Field(description=f["desc"], **ge, **le))
    return create_model("CommentMetrics", **fields)


def schema_matrix(features: list) -> str:
    """Render the feature config as a schema summary for the prompt."""
    lines = [f'  "{f["name"]}": "{f["desc"]}"' for f in features]
    return "Expected Output Schema Matrix:\n{\n" + ",\n".join(lines) + "\n}"


def _full_id(data: dict[str, Any], fallback_prefix: str) -> str | None:
    name = data.get("name")
    if name:
        return name
    raw_id = data.get("id")
    if raw_id:
        return f"{fallback_prefix}_{raw_id}"
    return None


def _iter_comment_nodes(comment_listing: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for child in comment_listing.get("data", {}).get("children", []):
        if child.get("kind") == "more":
            continue
        if child.get("kind") != "t1":
            continue
        comment_data = child.get("data", {})
        nodes.append(comment_data)
        replies = comment_data.get("replies")
        if isinstance(replies, dict):
            nodes.extend(_iter_comment_nodes(replies))
    return nodes


def build_created_utc_lookup(raw_dir: Path) -> dict[str, int]:
    """Map Reddit object ids (t1_/t3_) to created_utc from locally saved .json files."""
    lookup: dict[str, int] = {}
    if not raw_dir.exists():
        return lookup

    for path in sorted(raw_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                reddit_json = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {path.name} for date lookup: {e}")
            continue

        if not isinstance(reddit_json, list) or not reddit_json:
            continue

        post_children = reddit_json[0].get("data", {}).get("children", [])
        if post_children and post_children[0].get("kind") == "t3":
            post_data = post_children[0]["data"]
            post_id = _full_id(post_data, "t3")
            created_utc = post_data.get("created_utc")
            if post_id and created_utc is not None:
                lookup[post_id] = int(created_utc)

        if len(reddit_json) < 2:
            continue

        comment_listing = reddit_json[1]
        if not isinstance(comment_listing, dict):
            continue

        for comment_data in _iter_comment_nodes(comment_listing):
            comment_id = _full_id(comment_data, "t1")
            created_utc = comment_data.get("created_utc")
            if comment_id and created_utc is not None:
                lookup[comment_id] = int(created_utc)

    return lookup


def load_comments_from_csv(source_csv_path: Path, raw_dir: Path) -> list[dict[str, Any]]:
    """Load comment rows from the extracted Reddit CSV."""
    df = pd.read_csv(source_csv_path)
    date_lookup = build_created_utc_lookup(raw_dir)

    entries: list[dict[str, Any]] = []
    skipped = 0

    for _, row in df.iterrows():
        comment = str(row.get("comment", "")).strip()
        if not comment or comment in {"[deleted]", "[removed]"}:
            continue

        if "date" in df.columns and pd.notna(row["date"]):
            date = int(row["date"])
        elif "created_utc" in df.columns and pd.notna(row["created_utc"]):
            date = int(row["created_utc"])
        else:
            reddit_id = row.get("id")
            date = date_lookup.get(reddit_id) if pd.notna(reddit_id) else None

        if date is None:
            skipped += 1
            continue

        entries.append({
            "comment": comment,
            "date": date,
            "score": row["score"],
            "upvote_ratio": row["upvote_ratio"],
        })

    if skipped:
        print(f"Warning: skipped {skipped} row(s) with no date (check --raw-dir).")

    return entries


def ensure_model(client: Client, model: str):
    """Pull the model if it is not available locally."""
    local = {m.model for m in client.list().models}
    if model not in local:
        print(f"Pulling {model} (first run)...")
        client.pull(model)


def analyze_single_comment(client: Client, node: dict, model: str, system: str, schema) -> Optional[dict]:
    """Analyze a single comment and validate the model output."""
    try:
        response = client.generate(
            model=model,
            system=system,
            prompt=f"Analyze the financial context of this text:\n\n\"{node['comment']}\"",
            format=schema.model_json_schema(),
            options={"temperature": 0.0, "top_p": 0.1, "seed": 42}
        )

        metrics = schema.model_validate_json(response['response'])

        return {
            "comment": node["comment"],
            "date": node["date"],
            "score": node["score"],
            "upvote_ratio": node["upvote_ratio"],
            **metrics.model_dump()
        }

    except Exception as e:
        print(f"Failed to process comment parsing matrix: {e}")
        return None


def run_pipeline(
    source_csv_path: Path,
    target_csv_path: Path,
    model: str,
    system_prompt_path: Path,
    features_path: Path,
    raw_dir: Path,
):
    """Run the full comment analysis pipeline."""
    if not source_csv_path.exists():
        print(f"CRITICAL ERROR: Input database '{source_csv_path}' not found.")
        return

    raw_data = load_comments_from_csv(source_csv_path, raw_dir)
    if not raw_data:
        print("CRITICAL ERROR: No comment rows found in input CSV.")
        return

    with open(system_prompt_path, "r", encoding="utf-8") as f:
        instructions = f.read()
    with open(features_path, "r", encoding="utf-8") as f:
        features = yaml.safe_load(f)

    schema = build_schema(features)
    system = f"{instructions}\n{schema_matrix(features)}"

    client = Client()
    ensure_model(client, model)
    clean_rows = []

    print(f"Processing {len(raw_data)} comments one at a time...")
    for i, entry in enumerate(raw_data, start=1):
        print(f"Processing comment {i}/{len(raw_data)}...")
        result = analyze_single_comment(client, entry, model, system, schema)
        if result is not None:
            clean_rows.append(result)

    if not clean_rows:
        print("Pipeline aborted: No data rows were successfully analyzed.")
        return

    new_df = pd.DataFrame(clean_rows)
    target_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if target_csv_path.exists():
        new_df.to_csv(target_csv_path, mode='a', header=False, index=False)
        print(f"Success: Appended {len(clean_rows)} entries directly onto existing tracking structure at '{target_csv_path}'.")
    else:
        new_df.to_csv(target_csv_path, mode='w', header=True, index=False)
        print(f"Success: Created fresh repository asset profile. Exported clean metrics straight to '{target_csv_path}'.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Score reddit comments with a local LLM.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help="Reddit .json source folder used to resolve created_utc when the CSV has no date column.")
    p.add_argument("--model", choices=MODELS, default=MODELS[0])
    p.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    args = p.parse_args()

    run_pipeline(args.input, args.output, args.model, args.system_prompt, args.features, args.raw_dir)
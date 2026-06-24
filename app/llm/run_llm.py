import argparse
import json
import os
from typing import Optional
import pandas as pd
import yaml
from pydantic import Field, create_model
from ollama import Client

MODELS = ["gemma4:e4b", "gemma4:12b"]  # e4b: ~8GB VRAM, 12b: ~16GB VRAM
TYPES = {"int": int, "str": str, "float": float}


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


def run_pipeline(source_json_path: str, target_csv_path: str, model: str, system_prompt_path: str, features_path: str):
    """Run the full comment analysis pipeline."""
    if not os.path.exists(source_json_path):
        print(f"CRITICAL ERROR: Input database '{source_json_path}' not found.")
        return

    with open(source_json_path, "r") as f:
        raw_data = json.load(f)
    with open(system_prompt_path, "r") as f:
        instructions = f.read()
    with open(features_path, "r") as f:
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

    if os.path.exists(target_csv_path):
        new_df.to_csv(target_csv_path, mode='a', header=False, index=False)
        print(f"Success: Appended {len(clean_rows)} entries directly onto existing tracking structure at '{target_csv_path}'.")
    else:
        new_df.to_csv(target_csv_path, mode='w', header=True, index=False)
        print(f"Success: Created fresh repository asset profile. Exported clean metrics straight to '{target_csv_path}'.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Score reddit comments with a local LLM.")
    p.add_argument("--input", default="raw data (placeholder)/test_fakecomments.json")
    p.add_argument("--output", default="processed data (placeholder)/llm_output_test.csv")
    p.add_argument("--model", choices=MODELS, default=MODELS[0])
    p.add_argument("--system-prompt", default="app/modeling/system_prompt.md")
    p.add_argument("--features", default="app/modeling/features.yaml")
    args = p.parse_args()

    run_pipeline(args.input, args.output, args.model, args.system_prompt, args.features)
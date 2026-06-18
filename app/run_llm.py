import json
import os
from typing import Optional
import pandas as pd
from pydantic import BaseModel, Field
from ollama import Client

# Pydantic schema
class CommentMetrics(BaseModel):
    ticker: str = Field(description="Uppercase asset ticker or 'NONE'")
    sentiment: int = Field(ge=-3, le=3)
    conviction_level: int = Field(ge=0, le=4)
    emotional_intensity: int = Field(ge=0, le=4)
    fear_vs_greed: int = Field(ge=-2, le=2)
    certainty_of_forecast: int = Field(ge=0, le=4)
    time_horizon: int = Field(ge=-1, le=4)
    argument_logic: int = Field(ge=0, le=3)
    tone_profile: int = Field(ge=-1, le=2)
    company_fundamentals: int = Field(ge=-2, le=2)
    technical_chart: int = Field(ge=-2, le=2)
    suggested_action_aggression: int = Field(ge=0, le=3)


# Generate Ollama response
def analyze_single_comment(client: Client, node: dict) -> Optional[dict]:
    try:
        response = client.generate( # Call model
            model='gemma-sentiment-numeric',
            prompt=f"Analyze the financial context of this text:\n\n\"{node['comment']}\"",
            format=CommentMetrics.model_json_schema(),
            options={"temperature": 0.0}
        )

        metrics = CommentMetrics.model_validate_json(response['response']) # Validate the output against pydantic schema

        return { # Parse non LLM fields separately
            "comment": node["comment"],
            "date": node["date"],
            "score": node["score"],
            "upvote_ratio": node["upvote_ratio"],
            **metrics.model_dump()
        }

    except Exception as e:
        print(f"Failed to process comment parsing matrix: {e}")
        return None


# Run the pipeline
def run_pipeline(source_json_path: str, target_csv_path: str):
    if not os.path.exists(source_json_path):
        print(f"CRITICAL ERROR: Input database '{source_json_path}' not found.")
        return

    with open(source_json_path, "r") as f:
        raw_data = json.load(f)

    client = Client()
    clean_rows = []

    print(f"Processing {len(raw_data)} comments one at a time...")
    for i, entry in enumerate(raw_data, start=1):
        print(f"Processing comment {i}/{len(raw_data)}...")
        result = analyze_single_comment(client, entry)
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


# Change input and output files here
if __name__ == "__main__":
    INPUT_FILE = "raw data (placeholder)/test_fakecomments.json"
    OUTPUT_CSV = "processed data (placeholder)/llm_output_test.csv"

    run_pipeline(INPUT_FILE, OUTPUT_CSV)

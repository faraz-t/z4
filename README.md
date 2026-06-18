# Z4 Finance

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Running the Application

```bash
# Example: latest stock data
python3 -m app.main AAPL

# Example: specific date
python3 -m app.main TSLA --date 2024-07-13

# Example: reddit demo
python3 -m app.reddit
```

Note: PRAW uses Reddit API credentials to access Reddit data. Replace the placeholder data with your actual credentials locally. 

## LLM Comment Analysis

### Setup

1. Install and start Ollama (the script auto-pulls the model on first run)

### Usage

```bash
# Defaults: gemma4:e4b (~8GB VRAM), test set, app/system_prompt.md
python3 -m app.run_llm

# Larger model (~16GB VRAM, e.g. RTX 5080) and a custom prompt
python3 -m app.run_llm --model gemma4:12b --system-prompt app/system_prompt.md \
  --input "raw data (placeholder)/train_fakecomments.json" \
  --output "processed data (placeholder)/llm_output_train.csv"
```

Output fields live in `app/features.yaml` (one entry = one validator + prompt line + CSV
column). `app/system_prompt.md` holds the model's instructions/tone. The script appends to an existing output file if it exists, otherwise it creates a new one.
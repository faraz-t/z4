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

1. Install and start Ollama
2. Pull the base model

```
ollama pull gemma4:e4b

ollama create gemma-sentiment-numeric -f app/Modelfile
```

### Usage

`python3 -m app.run_llm`

Change input/output files in run_llm.py
It will attempt to append new rows to an existing file first.
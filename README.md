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

# Example: Reddit demo
python3 -m app.reddit_demo
```

The Reddit demo in `app/reddit_demo.py` uses PRAW. Replace the placeholder API values in that file with your own Reddit app credentials before running it.
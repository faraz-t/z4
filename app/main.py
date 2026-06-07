from __future__ import annotations

import argparse

import yfinance as yf

def get_latest_close(ticker_symbol: str) -> float:
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period="5d")

    if history.empty:
        raise ValueError(f"No market data found for {ticker_symbol}")

    latest_close = history["Close"].dropna().iloc[-1]
    return float(latest_close)

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a stock price from yfinance.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker symbol")
    args = parser.parse_args()

    latest_close = get_latest_close(args.ticker.upper())
    print(f"{args.ticker.upper()}: {latest_close:.2f}")

if __name__ == "__main__":
    main()
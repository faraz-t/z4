from __future__ import annotations

import argparse
from datetime import date, timedelta

import yfinance as yf

def get_latest_close(ticker_symbol: str) -> float:
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period="5d")

    if history.empty:
        raise ValueError(f"No market data found for {ticker_symbol}")

    latest_close = history["Close"].dropna().iloc[-1]
    return float(latest_close)


def get_close_for_date(ticker_symbol: str, requested_date: date, max_lookback: int = 7) -> tuple[float, date]:
    """Return (close, used_date) for the requested date or nearest previous trading date up to max_lookback days."""
    ticker = yf.Ticker(ticker_symbol)

    for days_back in range(0, max_lookback + 1):
        d = requested_date - timedelta(days=days_back)
        start_str = d.isoformat()
        end_str = (d + timedelta(days=1)).isoformat()
        history = ticker.history(start=start_str, end=end_str)
        if not history.empty and "Close" in history:
            val = float(history["Close"].dropna().iloc[-1])
            return val, d

    raise ValueError(f"No market data found for {ticker_symbol} within {max_lookback} days of {requested_date.isoformat()}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a stock price from yfinance.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--date", "-d", dest="date", help="Date in YYYY-MM-DD format to fetch (optional)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    if args.date:
        try:
            requested = date.fromisoformat(args.date)
        except ValueError:
            raise SystemExit(f"Invalid date format: {args.date}. Use YYYY-MM-DD")

        try:
            latest_close, used_date = get_close_for_date(ticker, requested)
            if used_date != requested:
                print(f"Note: no data for {requested.isoformat()}, using nearest previous trading date {used_date.isoformat()}")
        except ValueError as exc:
            raise SystemExit(str(exc))
    else:
        latest_close = get_latest_close(ticker)

    print(f"{ticker}: {latest_close:.2f}")

if __name__ == "__main__":
    main()
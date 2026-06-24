from __future__ import annotations

from datetime import date, timedelta
import yfinance as yf
import numpy as np


def get_close_for_date(ticker_symbol: str, requested_date: date, max_lookback: int = 7,) -> float:
    """Fetch close price for a given date with fallback lookback window."""

    ticker = yf.Ticker(ticker_symbol)

    for offset in range(max_lookback + 1):
        d = requested_date - timedelta(days=offset)

        hist = ticker.history(
            start=d.isoformat(),
            end=(d + timedelta(days=1)).isoformat(),
        )

        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])

    raise ValueError(f"No data found for {ticker_symbol}")


def compute_returns(ticker: str,base_date: date,) -> tuple[float, float, float, float]:
    """Returns: price0, ret7, ret14, ret30"""

    price0 = get_close_for_date(ticker, base_date)

    price7 = get_close_for_date(ticker, base_date + timedelta(days=7))
    price14 = get_close_for_date(ticker, base_date + timedelta(days=14))
    price30 = get_close_for_date(ticker, base_date + timedelta(days=30))

    ret7 = (price7 - price0) / price0
    ret14 = (price14 - price0) / price0
    ret30 = (price30 - price0) / price0

    return price0, ret7, ret14, ret30
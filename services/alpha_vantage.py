"""
FinSight AI — Data Service Layer
Primary:   yfinance  (free, unlimited, real-time)
Secondary: Alpha Vantage (premium, when API key is set)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────
# 1. yfinance  (primary — always works, no rate limits)
# ──────────────────────────────────────────────────────────────
def get_stock_data_yfinance(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetch OHLCV data via yfinance (Yahoo Finance).
    period options: 1mo, 3mo, 6mo, 1y, 2y, 5y
    Returns DataFrame with lowercase columns: open, high, low, close, volume
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(period=period)
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })[["open", "high", "low", "close", "volume"]]

        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index()

    except Exception as e:
        raise ConnectionError(f"yfinance error for '{symbol}': {e}")


# ──────────────────────────────────────────────────────────────
# 2. Alpha Vantage (secondary — requires API key)
# ──────────────────────────────────────────────────────────────
def get_stock_data_alphavantage(symbol: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Alpha Vantage TIME_SERIES_DAILY.
    Requires ALPHA_VANTAGE_API_KEY in .env
    Free tier: 25 requests/day, 5/min
    """
    import requests
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set — using yfinance instead")

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol.upper(),
        "apikey": API_KEY,
        "outputsize": "compact"
    }
    try:
        r = requests.get("https://www.alphavantage.co/query", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        if "Error Message" in data:
            raise ValueError(f"Invalid symbol: {symbol}")
        if "Note" in data or "Information" in data:
            raise ValueError("Alpha Vantage rate limit reached (25 req/day on free tier)")

        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(ts, orient="index").rename(columns={
            "1. open": "open", "2. high": "high",
            "3. low": "low", "4. close": "close", "5. volume": "volume"
        }).astype(float)
        df.index = pd.to_datetime(df.index)
        return df.sort_index().iloc[-100:]

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Network error: {e}")


# ──────────────────────────────────────────────────────────────
# 3. Public entry point  (yfinance first, AV as fallback)
# ──────────────────────────────────────────────────────────────
def get_stock_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """
    Primary: yfinance  (always used — free & unlimited)
    Falls back to Alpha Vantage only if yfinance fails.
    """
    try:
        df = get_stock_data_yfinance(symbol, period)
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback to Alpha Vantage
    return get_stock_data_alphavantage(symbol)


# ──────────────────────────────────────────────────────────────
# 4. Company info helper
# ──────────────────────────────────────────────────────────────
def get_company_info(symbol: str) -> dict:
    """Fetch company metadata (name, sector, market cap) via yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol.upper()).info
        return {
            "name":        info.get("longName", symbol.upper()),
            "sector":      info.get("sector", "N/A"),
            "industry":    info.get("industry", "N/A"),
            "market_cap":  info.get("marketCap", "N/A"),
            "country":     info.get("country", "N/A"),
            "website":     info.get("website", "N/A"),
            "description": (info.get("longBusinessSummary", "") or "")[:300]
        }
    except Exception:
        return {"name": symbol.upper(), "sector": "N/A"}
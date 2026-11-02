"""
FinSight AI — Utility & Helper Functions
"""
import numpy as np
import pandas as pd
from typing import Tuple, Union


def normalize_series(series: pd.Series) -> Tuple[pd.Series, float, float]:
    """
    Min-Max normalizes a pandas Series into [0, 1].
    Returns normalized series, min value, and max value.
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0.0, min_val, max_val
    normalized = (series - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def denormalize_series(normalized: Union[pd.Series, np.ndarray], min_val: float, max_val: float) -> np.ndarray:
    """Reverses min-max normalization."""
    return np.array(normalized) * (max_val - min_val) + min_val


def format_currency(value: float) -> str:
    """Formats float as $X,XXX.XX"""
    try:
        return f"${value:,.2f}"
    except (ValueError, TypeError):
        return "$0.00"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Formats float as percentage string."""
    try:
        return f"{value * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return "0.00%"


def calculate_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Computes Simple Moving Average (SMA)."""
    return df["close"].rolling(window=window).mean()


def calculate_ema(df: pd.DataFrame, span: int = 20) -> pd.Series:
    """Computes Exponential Moving Average (EMA)."""
    return df["close"].ewm(span=span, adjust=False).mean()

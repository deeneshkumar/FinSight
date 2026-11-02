"""
Backward-compatible wrapper for risk calculator.
Redirects to services.risk_calculator.
"""
from services.risk_calculator import (
    calculate_sharpe,
    calculate_volatility,
    calculate_var,
    calculate_max_drawdown,
    calculate_beta,
    classify_risk,
    full_risk_report,
)

__all__ = [
    "calculate_sharpe",
    "calculate_volatility",
    "calculate_var",
    "calculate_max_drawdown",
    "calculate_beta",
    "classify_risk",
    "full_risk_report",
]

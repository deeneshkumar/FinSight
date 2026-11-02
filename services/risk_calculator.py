"""
Risk Calculator for FinSight AI
Implements financial risk metrics used by portfolio managers:
  - Sharpe Ratio       : Risk-adjusted return
  - Volatility         : Annualized standard deviation of returns
  - Beta               : Sensitivity vs. S&P 500 (market)
  - VaR (95%)          : Value at Risk — worst expected loss in 95% of days
  - Max Drawdown       : Largest peak-to-trough decline
  - Risk Classification: Low / Moderate / High based on composite score
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.05   # ~5% annual (US 10Y Treasury proxy)
RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR


# ─────────────────────────────────────────────────────────────
# 1. Returns
# ─────────────────────────────────────────────────────────────
def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Daily percentage returns from close prices."""
    return df["close"].pct_change().dropna()


# ─────────────────────────────────────────────────────────────
# 2. Sharpe Ratio
# ─────────────────────────────────────────────────────────────
def calculate_sharpe(df: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE_DAILY) -> float:
    """
    Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev × √252
    
    Interpretation:
      > 1.0  → Good
      > 2.0  → Very Good
      > 3.0  → Excellent
      < 0    → Negative risk-adjusted return
    """
    returns = compute_returns(df)
    excess = returns - risk_free_rate
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std())


# ─────────────────────────────────────────────────────────────
# 3. Volatility (Annualized)
# ─────────────────────────────────────────────────────────────
def calculate_volatility(df: pd.DataFrame) -> float:
    """
    Annualized volatility = daily_std × √252
    Higher volatility → higher risk.
    """
    returns = compute_returns(df)
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


# ─────────────────────────────────────────────────────────────
# 4. Value at Risk (VaR) — Historical Method
# ─────────────────────────────────────────────────────────────
def calculate_var(df: pd.DataFrame, confidence: float = 0.95) -> float:
    """
    Historical VaR at 95% confidence.
    Means: On 95% of trading days, losses won't exceed this %.
    Returns the loss as a positive number (e.g., 0.03 = 3% max expected loss).
    """
    returns = compute_returns(df)
    return float(abs(np.percentile(returns, (1 - confidence) * 100)))


# ─────────────────────────────────────────────────────────────
# 5. Maximum Drawdown
# ─────────────────────────────────────────────────────────────
def calculate_max_drawdown(df: pd.DataFrame) -> float:
    """
    Max Drawdown = (Peak Price - Trough Price) / Peak Price
    Measures worst historical loss from a peak.
    """
    prices = df["close"]
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return float(drawdown.min())


# ─────────────────────────────────────────────────────────────
# 6. Beta (vs S&P 500 / market proxy)
# ─────────────────────────────────────────────────────────────
def calculate_beta(stock_df: pd.DataFrame, market_df: pd.DataFrame = None) -> float:
    """
    Beta = Cov(stock, market) / Var(market)
    
    If market data not provided, uses synthetic S&P 500 proxy
    (random walk with 10% annual return, used for demo).
    
    Beta > 1 → more volatile than market
    Beta < 1 → less volatile than market
    Beta = 1 → moves with market
    """
    stock_returns = compute_returns(stock_df)

    if market_df is None or market_df.empty:
        # Synthetic market proxy for demo
        np.random.seed(0)
        market_returns = pd.Series(
            np.random.normal(0.0004, 0.01, len(stock_returns)),
            index=stock_returns.index
        )
    else:
        market_returns = compute_returns(market_df).reindex(stock_returns.index).dropna()
        stock_returns = stock_returns.reindex(market_returns.index).dropna()

    if len(stock_returns) < 2:
        return 1.0

    cov_matrix = np.cov(stock_returns, market_returns)
    return float(cov_matrix[0, 1] / cov_matrix[1, 1])


# ─────────────────────────────────────────────────────────────
# 7. Composite Risk Score & Classification
# ─────────────────────────────────────────────────────────────
def classify_risk(volatility: float, var: float, sharpe: float) -> tuple:
    """
    Composite risk score [0-10] based on:
      - Volatility (annualized)
      - VaR (95%)
      - Sharpe Ratio (inversely contributes to risk)
    
    Returns: (score: float, label: str, color: str)
    """
    # Normalize each metric to [0, 10]
    vol_score = min(volatility / 0.6, 1.0) * 10   # 60% annual vol = max risk
    var_score = min(var / 0.05, 1.0) * 10          # 5% daily VaR = max risk
    sharpe_penalty = max(0, (2.0 - sharpe) / 2.0) * 5  # Penalize low Sharpe

    composite = (vol_score * 0.4 + var_score * 0.4 + sharpe_penalty * 0.2)
    composite = round(min(composite, 10.0), 2)

    if composite < 3.5:
        return composite, "🟢 Low Risk", "#22c55e"
    elif composite < 6.5:
        return composite, "🟡 Moderate Risk", "#eab308"
    else:
        return composite, "🔴 High Risk", "#ef4444"


# ─────────────────────────────────────────────────────────────
# 8. Full Risk Report
# ─────────────────────────────────────────────────────────────
def full_risk_report(df: pd.DataFrame, symbol: str = "", risk_tolerance: int = 5) -> dict:
    """
    Runs all risk metrics and returns a unified report dict.
    risk_tolerance: 1 (conservative) to 10 (aggressive)
    """
    sharpe = calculate_sharpe(df)
    volatility = calculate_volatility(df)
    var_95 = calculate_var(df)
    max_dd = calculate_max_drawdown(df)
    beta = calculate_beta(df)
    risk_score, risk_label, risk_color = classify_risk(volatility, var_95, sharpe)

    # Investment advice based on risk score vs user tolerance
    adjusted_score = risk_score - (risk_tolerance - 5) * 0.5
    if adjusted_score < 3:
        recommendation = "✅ Strong Buy — Risk aligns with your tolerance"
    elif adjusted_score < 5:
        recommendation = "🔵 Hold / Accumulate — Moderate risk, monitor closely"
    elif adjusted_score < 7:
        recommendation = "⚠️ Caution — Risk exceeds your tolerance slightly"
    else:
        recommendation = "🔴 Reduce Exposure — High risk for your profile"

    return {
        "symbol": symbol.upper(),
        "last_price": round(float(df["close"].iloc[-1]), 2),
        "sharpe_ratio": round(sharpe, 3),
        "annualized_volatility": f"{round(volatility * 100, 2)}%",
        "value_at_risk_95": f"{round(var_95 * 100, 3)}%",
        "max_drawdown": f"{round(max_dd * 100, 2)}%",
        "beta": round(beta, 3),
        "risk_score": risk_score,
        "risk_label": risk_label,
        "recommendation": recommendation
    }

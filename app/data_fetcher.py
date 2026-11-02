"""
Backward-compatible wrapper for data fetching.
Redirects to services.alpha_vantage.
"""
from services.alpha_vantage import get_stock_data as fetch_stock_data, get_stock_data, get_company_info

__all__ = ["fetch_stock_data", "get_stock_data", "get_company_info"]

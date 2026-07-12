from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# ══════════════════════════════════════════════════════════════════════════════
# DUAL DATA SOURCE STRATEGY (Option B Implementation)
# ──────────────────────────────────────────────────────────────────────────────
# • yfinance:  Used for TRAINING data — free, unlimited, ~10 years history per ticker
#   (~7,500 rows/ticker × 30 tickers = ~225,000 training rows)
# • Polygon:   Used for LIVE price lookups in the web app and financials features
#   (Polygon free tier still provides real-time quotes and quarterly financials)
#
# This hybrid approach maximizes training data volume (the real bottleneck) while
# keeping the live app responsive with Polygon's premium quote service.
# ══════════════════════════════════════════════════════════════════════════════

# ── API Keys ──────────────────────────────────────────────────────────────────
# Load from environment variables only. Set these in your .env file.
# Never hardcode keys here — if this file reaches a public repo, the key
# will be compromised. See .env.example for the required variable names.
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()

if not POLYGON_API_KEY:
    raise EnvironmentError(
        "POLYGON_API_KEY is not set. "
        "Create a .env file with POLYGON_API_KEY=your_key_here "
        "and load it before starting the app."
    )


def _polygon_request(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"https://api.polygon.io{path}"
    payload = {"apiKey": POLYGON_API_KEY}
    if params:
        payload.update(params)
    response = requests.get(url, params=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_yfinance_price_history(ticker: str, days: int = 2500) -> List[Dict[str, Any]]:
    """
    Fetches historical daily bars using yfinance (free, no rate limit).
    Returns up to ~10 years of daily data per ticker.
    This is used for model TRAINING only.
    Returns data in same format as Polygon for compatibility.
    """
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        
        # Download data from Yahoo Finance
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if df.empty:
            print(f"No data from yfinance for {ticker}")
            return []

        # Flatten MultiIndex columns if present (yfinance 1.5+ adds a ticker level)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Drop duplicate column names that can appear after flattening
        df = df.loc[:, ~df.columns.duplicated()]

        # Convert to Polygon-compatible format
        results = []
        for date, row in df.iterrows():
            results.append({
                "t": int(pd.Timestamp(date).timestamp() * 1000),
                "o": float(row["Open"]),
                "h": float(row["High"]),
                "l": float(row["Low"]),
                "c": float(row["Close"]),
                "v": int(row["Volume"]),
            })
        
        return results
    except Exception as e:
        print(f"Error fetching yfinance history for {ticker}: {e}")
        return []


def fetch_yfinance_financials(ticker: str) -> pd.DataFrame:
    """
    Fetches quarterly financial statements from yfinance — free, no rate limits.
    Returns a tidy DataFrame with one row per quarter.
    Used for model TRAINING only.
    """
    try:
        t = yf.Ticker(ticker)

        income   = t.quarterly_income_stmt
        balance  = t.quarterly_balance_sheet
        cashflow = t.quarterly_cashflow

        if income is None or income.empty:
            return pd.DataFrame()

        def _get_val(df, col, *keys):
            """Safely extract a float value from a DataFrame at a specific column."""
            if df is None or df.empty:
                return np.nan
            for k in keys:
                if k in df.index:
                    try:
                        val = df.at[k, col]
                        if pd.notna(val):
                            return float(val)
                    except Exception:
                        pass
            return np.nan

        rows = []
        for col in income.columns:
            revenue     = _get_val(income,   col, "Total Revenue")
            net_income  = _get_val(income,   col, "Net Income")
            assets      = _get_val(balance,  col, "Total Assets")
            liabilities = _get_val(balance,  col,
                                   "Total Liabilities Net Minority Interest",
                                   "Total Liabilities")
            cash_flow   = _get_val(cashflow, col,
                                   "Operating Cash Flow",
                                   "Cash Flow From Continuing Operating Activities")

            rows.append({
                "date":        pd.Timestamp(col).normalize(),
                "revenue":     revenue,
                "assets":      assets,
                "liabilities": liabilities,
                "cash_flow":   cash_flow,
                "net_income":  net_income,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        df["date"] = df["date"].astype("datetime64[us]")

        df["revenue_growth_qoq"] = df["revenue"].pct_change().fillna(0).clip(-2, 2)
        df["asset_growth_qoq"]   = df["assets"].pct_change().fillna(0).clip(-2, 2)
        df["liability_ratio"]    = (df["liabilities"] / df["assets"]).clip(0, 5).fillna(0)
        df["cash_flow_to_assets"]= (df["cash_flow"]   / df["assets"]).clip(-5, 5).fillna(0)

        return df

    except Exception as e:
        print(f"Error fetching yfinance financials for {ticker}: {e}")
        return pd.DataFrame()


def fetch_polygon_price_history(ticker: str, days: int = 1250) -> List[Dict[str, Any]]:
    """Fetches historical daily bars. 1250 days corresponds to roughly 5 years of data."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=days * 1.45)  # Padding for weekends/holidays
    payload = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
    }
    try:
        data = _polygon_request(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start:%Y-%m-%d}/{end:%Y-%m-%d}",
            payload,
        )
        return data.get("results", []) if isinstance(data, dict) else []
    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        return []


def fetch_polygon_financials(ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Fetches up to 20 quarters of financial statements."""
    try:
        data = _polygon_request(
            "/vX/reference/financials",
            {
                "ticker": ticker,
                "timeframe": "quarterly",
                "limit": limit,
            },
        )
        return data.get("results", []) if isinstance(data, dict) else []
    except Exception as e:
        print(f"Error fetching financials for {ticker}: {e}")
        return []


# ── Liability thresholds ──────────────────────────────────────────────────────
# Defined BEFORE fetch_sector_liability_threshold() which references them.

SECTOR_LIABILITY_THRESHOLD = {
    # Finance — high leverage is the normal business model
    "JPM": 0.92, "BAC": 0.92, "GS": 0.92, "C": 0.92, "V": 0.75, "MA": 0.75,
    # Utilities / telecoms — capital-intensive, permanently debt-heavy
    "T": 0.85,
    # Energy — cyclically leveraged
    "XOM": 0.75, "CVX": 0.72,
    # Consumer staples — moderate leverage normal
    "KO": 0.78, "PG": 0.72, "WMT": 0.72, "PEP": 0.75,
    # Industrials
    "BA": 0.85,
}
DEFAULT_LIABILITY_THRESHOLD = 0.65

# Sector-keyword-to-threshold mapping for dynamic Polygon lookups
SECTOR_THRESHOLD_MAP = {
    "financial services": 0.92,
    "financials":         0.92,
    "banking":            0.92,
    "utilities":          0.85,
    "energy":             0.75,
    "consumer staples":   0.75,
    "industrials":        0.80,
    "real estate":        0.88,
}


def fetch_sector_liability_threshold(ticker: str) -> float:
    """
    Returns the appropriate liability threshold for a ticker.
    Checks hardcoded dict first (no API call for known tickers), then
    falls back to a live Polygon sector lookup, then DEFAULT_LIABILITY_THRESHOLD.
    """
    if ticker in SECTOR_LIABILITY_THRESHOLD:
        return SECTOR_LIABILITY_THRESHOLD[ticker]
    try:
        data = _polygon_request(f"/v3/reference/tickers/{ticker}")
        sector = (
            data.get("results", {})
            .get("sic_description", "")
            .lower()
        )
        for key, threshold in SECTOR_THRESHOLD_MAP.items():
            if key in sector:
                return threshold
    except Exception:
        pass
    return DEFAULT_LIABILITY_THRESHOLD


def resolve_tradingview_symbol(ticker: str) -> str:
    normalized = (ticker or "").strip().upper()
    if not normalized:
        return "NASDAQ:AAPL"
    if ":" in normalized:
        return normalized

    exchange_map = {
        "AAPL": "NASDAQ:AAPL", "MSFT": "NASDAQ:MSFT", "AMZN": "NASDAQ:AMZN",
        "TSLA": "NASDAQ:TSLA", "NVDA": "NASDAQ:NVDA", "META": "NASDAQ:META",
        "GOOGL": "NASDAQ:GOOGL", "GOOG": "NASDAQ:GOOG", "AMD": "NASDAQ:AMD",
        "INTC": "NASDAQ:INTC", "NFLX": "NASDAQ:NFLX", "ORCL": "NASDAQ:ORCL",
        "PEP": "NASDAQ:PEP", "ADBE": "NASDAQ:ADBE", "CRM": "NASDAQ:CRM",
        "QCOM": "NASDAQ:QCOM", "AVGO": "NASDAQ:AVGO", "PYPL": "NASDAQ:PYPL",
        "IBM": "NYSE:IBM", "BA": "NYSE:BA", "DIS": "NYSE:DIS",
        "JPM": "NYSE:JPM", "V": "NYSE:V", "MA": "NYSE:MA",
        "PG": "NYSE:PG", "KO": "NYSE:KO", "WMT": "NYSE:WMT",
        "T": "NYSE:T", "XOM": "NYSE:XOM", "CVX": "NYSE:CVX",
        "LLY": "NYSE:LLY", "MRK": "NYSE:MRK", "ABBV": "NYSE:ABBV",
        "JNJ": "NYSE:JNJ", "PFE": "NYSE:PFE", "BAC": "NYSE:BAC",
        "GS": "NYSE:GS", "C": "NYSE:C", "SPY": "AMEX:SPY",
        "QQQ": "NASDAQ:QQQ", "BRK.B": "NYSE:BRK.B",
    }
    return exchange_map.get(normalized, normalized)
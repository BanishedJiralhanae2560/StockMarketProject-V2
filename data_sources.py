from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# ── API Keys ──────────────────────────────────────────────────────────────────
POLYGON_API_KEY       = os.getenv("POLYGON_API_KEY", "").strip()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()

if not POLYGON_API_KEY:
    raise EnvironmentError(
        "POLYGON_API_KEY is not set. "
        "Create a .env file with POLYGON_API_KEY=your_key_here "
        "and load it before starting the app."
    )


def _polygon_request(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url     = f"https://api.polygon.io{path}"
    payload = {"apiKey": POLYGON_API_KEY}
    if params:
        payload.update(params)
    response = requests.get(url, params=payload, timeout=15)
    response.raise_for_status()
    return response.json()


# ── yfinance — used for ALL training data (price + financials) ────────────────

def fetch_yfinance_price_history(ticker: str, days: int = 2500) -> List[Dict[str, Any]]:
    """
    Fetches historical daily OHLCV bars via yfinance (free, no rate limit).
    Returns data in Polygon-compatible format for drop-in compatibility.
    Used for model TRAINING only.
    """
    try:
        end   = datetime.utcnow().date()
        start = end - timedelta(days=days)

        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            print(f"No price data from yfinance for {ticker}")
            return []

        # yfinance 1.5+ returns MultiIndex columns — flatten to single level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]

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
    Fetches quarterly financial statements via yfinance (free, no rate limit).
    Returns a tidy DataFrame — one row per quarter — with computed growth metrics.
    Key names match yfinance 1.5.1 actual index labels confirmed by diagnostics.
    Used for model TRAINING only.
    """
    try:
        t        = yf.Ticker(ticker)
        income   = t.quarterly_income_stmt
        balance  = t.quarterly_balance_sheet
        cashflow = t.quarterly_cashflow

        if income is None or income.empty:
            return pd.DataFrame()

        def _get(df, col, *keys):
            """Safely extract a float scalar from df at (row_key, col)."""
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
            # Key names confirmed from yfinance 1.5.1 diagnostic output
            revenue     = _get(income, col,
                               "Total Revenue",
                               "Operating Revenue")
            net_income  = _get(income, col,
                               "Net Income",
                               "Net Income From Continuing Operation Net Minority Interest",
                               "Net Income From Continuing And Discontinued Operation")
            assets      = _get(balance, col,
                               "Total Assets")
            liabilities = _get(balance, col,
                               "Total Liabilities Net Minority Interest",
                               "Total Liabilities",
                               "Total Non Current Liabilities Net Minority Interest")
            cash_flow   = _get(cashflow, col,
                               "Operating Cash Flow",
                               "Cash Flow From Continuing Operating Activities",
                               "Net Cash Provided By Operating Activities")

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

        # Quarter-over-quarter growth — sorted oldest→newest so pct_change is correct
        df["revenue_growth_qoq"] = df["revenue"].pct_change().fillna(0).clip(-2, 2)
        df["asset_growth_qoq"]   = df["assets"].pct_change().fillna(0).clip(-2, 2)
        df["liability_ratio"]    = (df["liabilities"] / df["assets"]).clip(0, 5).fillna(0)
        df["cash_flow_to_assets"]= (df["cash_flow"]   / df["assets"]).clip(-5, 5).fillna(0)

        return df

    except Exception as e:
        print(f"Error fetching yfinance financials for {ticker}: {e}")
        return pd.DataFrame()


# ── Polygon — used for LIVE app only (real-time quotes + live financials) ─────

def fetch_polygon_price_history(ticker: str, days: int = 1250) -> List[Dict[str, Any]]:
    """Fetches historical daily bars from Polygon. Used by the live app."""
    end     = datetime.utcnow().date()
    start   = end - timedelta(days=days * 1.45)
    payload = {"adjusted": "true", "sort": "asc", "limit": 5000}
    try:
        data = _polygon_request(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start:%Y-%m-%d}/{end:%Y-%m-%d}",
            payload,
        )
        return data.get("results", []) if isinstance(data, dict) else []
    except Exception as e:
        print(f"Error fetching Polygon history for {ticker}: {e}")
        return []


def fetch_polygon_financials(ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Fetches quarterly financials from Polygon. Used by the live app."""
    try:
        data = _polygon_request(
            "/vX/reference/financials",
            {"ticker": ticker, "timeframe": "quarterly", "limit": limit},
        )
        return data.get("results", []) if isinstance(data, dict) else []
    except Exception as e:
        print(f"Error fetching Polygon financials for {ticker}: {e}")
        return []


# ── Liability thresholds — defined before the function that references them ───

SECTOR_LIABILITY_THRESHOLD = {
    "JPM": 0.92, "BAC": 0.92, "GS": 0.92, "C": 0.92, "V": 0.75, "MA": 0.75,
    "T":   0.85,
    "XOM": 0.75, "CVX": 0.72,
    "KO":  0.78, "PG":  0.72, "WMT": 0.72, "PEP": 0.75,
    "BA":  0.85,
}
DEFAULT_LIABILITY_THRESHOLD = 0.65

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
    if ticker in SECTOR_LIABILITY_THRESHOLD:
        return SECTOR_LIABILITY_THRESHOLD[ticker]
    try:
        data   = _polygon_request(f"/v3/reference/tickers/{ticker}")
        sector = data.get("results", {}).get("sic_description", "").lower()
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
        "IBM": "NYSE:IBM",   "BA":  "NYSE:BA",   "DIS": "NYSE:DIS",
        "JPM": "NYSE:JPM",   "V":   "NYSE:V",    "MA":  "NYSE:MA",
        "PG":  "NYSE:PG",    "KO":  "NYSE:KO",   "WMT": "NYSE:WMT",
        "T":   "NYSE:T",     "XOM": "NYSE:XOM",  "CVX": "NYSE:CVX",
        "LLY": "NYSE:LLY",  "MRK": "NYSE:MRK",  "ABBV":"NYSE:ABBV",
        "JNJ": "NYSE:JNJ",  "PFE": "NYSE:PFE",  "BAC": "NYSE:BAC",
        "GS":  "NYSE:GS",   "C":   "NYSE:C",    "SPY": "AMEX:SPY",
        "QQQ": "NASDAQ:QQQ","BRK.B":"NYSE:BRK.B",
    }
    return exchange_map.get(normalized, normalized)
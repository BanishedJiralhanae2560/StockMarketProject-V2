from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import joblib
import pandas as pd
import numpy as np
import requests
import uvicorn

from data_sources import (
    fetch_polygon_financials,
    fetch_polygon_price_history,
    resolve_tradingview_symbol,
    fetch_sector_liability_threshold,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── API key loaded from environment — never hardcoded ─────────────────────────
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
if not POLYGON_API_KEY:
    raise EnvironmentError(
        "POLYGON_API_KEY is not set. "
        "Create a .env file with POLYGON_API_KEY=your_key and load it before starting."
    )

MODEL_PATH       = Path("models/lightgbm_stock_signal.pkl")
DATA_PATH        = Path("data/apple_historical_prices.csv")
SAMPLE_DATA_PATH = Path("data/sample_training_data.csv")


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_pct(current: float, previous: float) -> float:
    if previous and previous != 0:
        return (current - previous) / previous
    return 0.0


def decode_value(value):
    if isinstance(value, dict):
        return value.get("value") or value.get("amount") or 0.0
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def load_model() -> Optional[object]:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def fetch_price_history(ticker: str, days: int = 100) -> List[Dict]:
    return fetch_polygon_price_history(ticker, days=days)


def load_local_price_data(ticker: str) -> Optional[pd.DataFrame]:
    if ticker == "AAPL" and DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        df = df.rename(columns={
            "Date": "date", "AAPL.Open": "open", "AAPL.High": "high",
            "AAPL.Low": "low", "AAPL.Close": "close",
            "AAPL.Volume": "volume", "AAPL.Adjusted": "adjusted",
        })
        return df.sort_values("date").reset_index(drop=True)
    if SAMPLE_DATA_PATH.exists():
        df = pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)
    return None


def build_fallback_result(ticker: str, df: pd.DataFrame) -> dict:
    """Used when Polygon API is unavailable and local CSV data exists."""
    if df.empty:
        return {
            'stock_name': ticker, 'current_price': 'N/A',
            'change': 0.0, 'change_percentage': 0.0, 'volume': 0,
            'buy_signal': False, 'buy_recommendation': 'Hold',
            'confidence': 'Low', 'risk_level': 'Medium', 'growth_factor': 0,
            'quarterly_data': [], 'revenue_growth': None, 'asset_growth': None,
            'sell_signal': False, 'sell_recommendation': 'Hold',
            'sell_pressure': 0.0, 'sell_confidence': 'Low',
            'liability_ratio': None, 'volume_anomaly_ratio': 1.0,
            'earnings_surprise': 0.0,
        }

    last       = df.iloc[-1]
    last_close = float(last['close'])
    closes     = df['close'].tolist()
    volumes    = df['volume'].tolist() if 'volume' in df.columns else []

    ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else last_close
    ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else last_close

    buy_signal  = ma20 > ma50 and last_close > ma50
    sell_signal = ma20 < ma50

    vol_avg   = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 1
    vol_ratio = float(np.clip(volumes[-1] / (vol_avg + 1), 0, 10)) if volumes else 1.0

    return {
        'stock_name':          ticker,
        'current_price':       round(last_close, 2),
        'change':              round(last_close - float(df.iloc[-2]['close']), 2) if len(df) >= 2 else 0.0,
        'change_percentage':   round(safe_pct(last_close, float(df.iloc[-2]['close'])) * 100, 2) if len(df) >= 2 else 0.0,
        'volume':              int(last['volume']) if 'volume' in last and not pd.isna(last['volume']) else 0,
        'volume_anomaly_ratio': vol_ratio,
        'quarterly_data':      [],
        'revenue_growth':      None,
        'asset_growth':        None,
        'growth_factor':       0,
        'buy_signal':          buy_signal,
        'buy_recommendation':  'Buy' if buy_signal else 'Hold',
        'confidence':          'Moderate' if buy_signal else 'Low',
        'risk_level':          'Medium',
        'sell_signal':         sell_signal,
        'sell_recommendation': 'Sell' if sell_signal else 'Hold',
        'sell_pressure':       7.0 if sell_signal else 2.0,
        'sell_confidence':     'Moderate' if sell_signal else 'Low',
        'liability_ratio':     None,
        'earnings_surprise':   0.0,
    }


# ── ML feature vector ──────────────────────────────────────────────────────────

def build_model_features(
    agg_data: Dict,
    quarterly_data: List[Dict],
    price_history: List[Dict],
) -> List[float]:
    """
    Builds a 14-element feature vector identical in order and computation
    to train_model.py build_features(). Any change here must be mirrored there.

    Feature order:
        price_change_pct, return_1d, return_5d, return_10d,
        volatility_10d, volume_avg_10d, volume_anomaly_ratio,
        revenue_growth_qoq, asset_growth_qoq, liability_ratio,
        cash_flow_to_assets, rsi_14, macd_hist, earnings_surprise
    """
    latest = quarterly_data[0] if quarterly_data else {}

    # Fundamental features (fractions, clipped same as training)
    revenue_growth    = float(latest.get("revenue_growth_qoq") or 0.0)
    asset_growth      = float(latest.get("asset_growth_qoq")   or 0.0)
    assets            = float(latest.get("assets")             or 0.0)
    liabilities       = float(latest.get("liabilities")        or 0.0)
    cash_flow_value   = float(latest.get("cash_flow_value")    or 0.0)
    earnings_surprise = float(latest.get("earnings_surprise",    0.0))

    liability_ratio    = (liabilities / assets)    if assets else 0.0
    cash_flow_to_assets= (cash_flow_value / assets) if assets else 0.0

    # Price history features
    ph_df = pd.DataFrame(price_history)
    if ph_df.empty or "c" not in ph_df.columns:
        return [0.0] * 14

    closes  = ph_df["c"].astype(float).tolist()
    volumes = ph_df["v"].astype(float).tolist()

    last_close    = closes[-1]   if closes            else float(agg_data.get("c", 0.0))
    prev_close_1d = closes[-2]   if len(closes) >= 2  else last_close
    close_5d      = closes[-6]   if len(closes) >= 6  else prev_close_1d
    close_10d     = closes[-11]  if len(closes) >= 11 else prev_close_1d

    return_1d  = safe_pct(last_close, prev_close_1d)
    return_5d  = safe_pct(last_close, close_5d)
    return_10d = safe_pct(last_close, close_10d)

    pct_changes    = [safe_pct(closes[i], closes[i - 1]) for i in range(1, len(closes))]
    volatility_10d = float(pd.Series(pct_changes).tail(10).std()) if len(pct_changes) >= 10 else 0.0

    volume_avg_10d      = float(pd.Series(volumes).tail(10).mean()) if len(volumes) >= 10 else 0.0
    current_volume      = float(volumes[-1]) if volumes else 0.0
    volume_anomaly_ratio= float(np.clip(current_volume / (volume_avg_10d + 1), 0, 10))

    price_change_pct = safe_pct(float(agg_data.get("c", 0.0)), float(agg_data.get("o", 0.0)))

    # RSI — Wilder's EWM (alpha=1/14), identical to train_model.py _wilder_rsi
    delta    = ph_df["c"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_14 = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

    # MACD — single ticker so no groupby needed
    ema12       = ph_df["c"].ewm(span=12, adjust=False).mean()
    ema26       = ph_df["c"].ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = float((macd_line - macd_signal).iloc[-1]) if len(macd_line) > 0 else 0.0

    return [
        price_change_pct,
        return_1d,
        return_5d,
        return_10d,
        volatility_10d,
        volume_avg_10d,
        volume_anomaly_ratio,
        float(np.clip(revenue_growth,     -2.0, 2.0)),
        float(np.clip(asset_growth,       -2.0, 2.0)),
        float(np.clip(liability_ratio,     0.0, 5.0)),
        float(np.clip(cash_flow_to_assets,-5.0, 5.0)),
        rsi_14,
        macd_hist,
        earnings_surprise,
    ]


# ── Model loaded once at startup ───────────────────────────────────────────────
model = load_model()


# ── Core data fetch ────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str) -> dict:
    base_url = "https://api.polygon.io/v2"
    result   = {}

    try:
        # Previous day aggregate
        agg_url  = f"{base_url}/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={POLYGON_API_KEY}"
        agg_data = {}
        try:
            agg_resp = requests.get(agg_url, timeout=10)
            agg_json = agg_resp.json()
            agg_data = agg_json.get("results", [{}])[0]
        except Exception:
            agg_data = {}

        if not agg_data or not isinstance(agg_data, dict):
            local = load_local_price_data(ticker)
            if local is not None:
                return build_fallback_result(ticker, local)
            raise ValueError("No price data available for ticker")

        # Price history (100 days to guarantee MA50 always has enough bars)
        price_history = fetch_price_history(ticker, days=100)

        # Compute volume anomaly here so it's available for result dict
        ph_df_vol = pd.DataFrame(price_history)
        if not ph_df_vol.empty and "v" in ph_df_vol.columns:
            vols          = ph_df_vol["v"].astype(float).tolist()
            vol_avg       = float(pd.Series(vols).tail(10).mean()) if len(vols) >= 10 else 0.0
            vol_current   = float(vols[-1]) if vols else 0.0
            vol_anomaly   = float(np.clip(vol_current / (vol_avg + 1), 0, 10))
        else:
            vol_anomaly = 1.0

        result["stock_name"]           = ticker
        result["chart_symbol"]         = resolve_tradingview_symbol(ticker)
        result["current_price"]        = agg_data.get("c", 0.0)
        result["change"]               = round(agg_data.get("c", 0.0) - agg_data.get("o", 0.0), 2)
        result["change_percentage"]    = round(safe_pct(agg_data.get("c", 0.0), agg_data.get("o", 0.0)) * 100, 2)
        result["volume"]               = agg_data.get("v", 0)
        result["volume_anomaly_ratio"] = vol_anomaly

        # Financials
        financial_results      = fetch_polygon_financials(ticker, limit=4)
        quarterly_data         = []
        revenue_growth_fraction= 0.0
        asset_growth_fraction  = 0.0

        for i, quarter in enumerate(financial_results):
            financials  = quarter.get("financials", {})
            income      = financials.get("income_statement", {})
            balance     = financials.get("balance_sheet", {})
            cf_stmt     = financials.get("cash_flow_statement", {})

            def _get(d, key):
                v = d.get(key)
                return v.get("value") if isinstance(v, dict) else v

            revenue    = _get(income,   "revenues")
            expenses   = _get(income,   "operating_expenses")
            assets     = _get(balance,  "assets")
            liabilities= _get(balance,  "liabilities")
            period     = (
                quarter.get("filing_date")
                or quarter.get("report_period")
                or f"Q{quarter.get('fiscal_period','')} {quarter.get('fiscal_year','')}"
            )
            cf_val = decode_value(
                cf_stmt.get("cash_flow")
                or cf_stmt.get("net_cash_flow")
                or cf_stmt.get("net_cash_from_operating_activities")
            )

            quarter_data = {
                "period": period, "revenue": revenue, "expenses": expenses,
                "assets": assets, "liabilities": liabilities,
                "cash_flow_value": cf_val,
                "revenue_growth_qoq": 0.0, "asset_growth_qoq": 0.0,
                "net_income": decode_value(_get(income, "net_income_loss") or _get(income, "net_income")),
            }
            quarterly_data.append(quarter_data)

            if i == 0:
                result["income_data"]    = income
                result["balance_sheet"]  = balance
                result["cash_flow"]      = cf_stmt
                result["cash_flow_value"]= cf_val

            if i > 0 and all(isinstance(v, (int, float)) for v in [
                quarterly_data[i - 1].get("revenue"), quarter_data.get("revenue"),
                quarterly_data[i - 1].get("assets"),  quarter_data.get("assets"),
            ]):
                newer_rev = quarterly_data[i - 1]["revenue"]
                older_rev = quarter_data["revenue"]
                newer_ast = quarterly_data[i - 1]["assets"]
                older_ast = quarter_data["assets"]

                if older_rev and newer_rev:
                    revenue_growth_fraction = (newer_rev - older_rev) / abs(older_rev)
                if older_ast and newer_ast:
                    asset_growth_fraction   = (newer_ast - older_ast) / abs(older_ast)

        if quarterly_data:
            # Store computed growth back so build_model_features reads real values
            quarterly_data[0]["revenue_growth_qoq"] = revenue_growth_fraction
            quarterly_data[0]["asset_growth_qoq"]   = asset_growth_fraction

            # Earnings surprise: did most recent quarter beat the prior one?
            ni = [q.get("net_income") for q in quarterly_data[:2] if q.get("net_income") is not None]
            quarterly_data[0]["earnings_surprise"] = (
                1.0 if len(ni) == 2 and ni[0] > ni[1] else
               -1.0 if len(ni) == 2 and ni[0] < ni[1] else
                0.0
            )

        result["quarterly_data"]  = quarterly_data
        result["revenue_growth"]  = round(revenue_growth_fraction * 100, 2)
        result["asset_growth"]    = round(asset_growth_fraction   * 100, 2)
        result["buy_probability"] = None
        result["model_used"]      = False

        # ── ML path ───────────────────────────────────────────────────────
        if model is not None and quarterly_data:
            feature_vector = build_model_features(agg_data, quarterly_data, price_history)
            try:
                probability = float(model.predict_proba([feature_vector])[0, 1])
            except Exception:
                probability = None

            if probability is not None:
                result["buy_probability"]    = round(probability * 100, 1)
                result["buy_signal"]         = probability >= 0.55
                result["buy_recommendation"] = "Buy" if result["buy_signal"] else "Hold"
                result["confidence"]         = (
                    "High"     if probability >= 0.70 else
                    "Moderate" if probability >= 0.55 else "Low"
                )
                result["risk_level"] = (
                    "Low"    if probability >= 0.60 else
                    "Medium" if probability >= 0.45 else "High"
                )
                raw_gf = (revenue_growth_fraction * 10) + (asset_growth_fraction * 10)
                result["growth_factor"] = round(max(0.0, min(10.0, raw_gf)), 1)
                result["model_used"]    = True

        # ── Fallback buy signal (MA crossover + fundamentals) ─────────────
        if not result["model_used"]:
            closes = [bar.get("c", 0.0) for bar in price_history if bar.get("c")]
            ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
            ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None

            ma_crossover   = ma20 is not None and ma50 is not None and ma20 > ma50
            fundamental_ok = revenue_growth_fraction > 0 and asset_growth_fraction > 0
            buy_signal     = ma_crossover and fundamental_ok
            growth_factor  = round(
                max(0.0, min(10.0,
                    (revenue_growth_fraction * 5) + (asset_growth_fraction * 5)
                )), 1
            )

            result["buy_signal"]         = buy_signal
            result["buy_recommendation"] = "Buy" if buy_signal else "Hold"
            result["confidence"]         = (
                "High"     if buy_signal and growth_factor >= 7 else
                "Moderate" if buy_signal else "Low"
            )
            result["risk_level"]   = (
                "Low"    if buy_signal and growth_factor >= 7 else
                "Medium" if buy_signal else "High"
            )
            result["growth_factor"] = growth_factor

        # ── Sell signal (death cross + fundamental pressure) ──────────────
        closes_all = [bar.get("c", 0.0) for bar in price_history if bar.get("c")]
        ma20_s = sum(closes_all[-20:]) / 20 if len(closes_all) >= 20 else None
        ma50_s = sum(closes_all[-50:]) / 50 if len(closes_all) >= 50 else None
        death_cross  = ma20_s is not None and ma50_s is not None and ma20_s < ma50_s
        sell_pressure= 0.0

        if revenue_growth_fraction < 0:          sell_pressure += 4.0
        if asset_growth_fraction   < 0:          sell_pressure += 2.0
        if result["change_percentage"] < -2:     sell_pressure += 2.0
        if death_cross:                          sell_pressure += 3.0

        latest_q    = quarterly_data[0] if quarterly_data else {}
        lat_assets  = latest_q.get("assets")
        lat_liab    = latest_q.get("liabilities")
        liability_ratio = None
        if lat_assets and lat_liab and lat_assets > 0:
            liability_ratio     = lat_liab / lat_assets
            liab_threshold      = fetch_sector_liability_threshold(ticker)
            if liability_ratio > liab_threshold:
                sell_pressure  += 2.0

        sell_pressure = round(min(10.0, sell_pressure), 1)
        # Death cross alone during consolidation shouldn't fire — require at least one other signal
        result["sell_signal"]         = sell_pressure >= 5.0 or (death_cross and sell_pressure >= 2.0)
        result["sell_recommendation"] = "Sell" if result["sell_signal"] else "Hold"
        result["sell_pressure"]       = sell_pressure
        result["sell_confidence"]     = (
            "High"     if sell_pressure >= 7.0 else
            "Moderate" if sell_pressure >= 5.0 else "Low"
        )
        result["liability_ratio"]   = round(liability_ratio, 2) if liability_ratio is not None else None
        result["earnings_surprise"] = quarterly_data[0].get("earnings_surprise", 0.0) if quarterly_data else 0.0

    except Exception as e:
        local = load_local_price_data(ticker)
        if local is not None:
            return build_fallback_result(ticker, local)
        result = {"error": str(e)}

    return result


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/data")
async def get_data(ticker: str = Query(default="AAPL")):
    return fetch_stock_data(ticker.upper())


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
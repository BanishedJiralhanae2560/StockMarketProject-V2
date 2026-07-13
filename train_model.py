from dotenv import load_dotenv
load_dotenv()

import time
import shutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier

from data_sources import fetch_yfinance_price_history, fetch_yfinance_financials

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "lightgbm_stock_signal.pkl"

TICKERS_POOL = [
    "AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "GOOGL", "AMD", "INTC", "NFLX",
    "ORCL", "PEP", "ADBE", "CRM", "QCOM", "AVGO", "IBM", "BA", "DIS", "JPM",
    "V", "MA", "PG", "KO", "WMT", "T", "XOM", "CVX", "LLY", "MRK"
]


def collect_bulk_api_data() -> pd.DataFrame:
    all_frames = []
    print(f"Starting bulk data collection for {len(TICKERS_POOL)} tickers using yfinance...")
    print("(yfinance provides free access to ~10 years of daily data per ticker)\n")

    for ticker in TICKERS_POOL:
        print(f"  Fetching: {ticker}")
        # Match the available quarterly financial window so most rows retain
        # real fundamental features instead of falling back to zeros.
        results = fetch_yfinance_price_history(ticker, days=548)  # ~18 months
        if not results:
            print(f"    No price data for {ticker}, skipping.")
            continue

        price_df = pd.DataFrame(results).rename(columns={
            "t": "date", "o": "open", "h": "high",
            "l": "low",  "c": "close", "v": "volume",
        })
        price_df["date"]   = pd.to_datetime(price_df["date"], unit="ms").dt.normalize().astype("datetime64[us]")
        price_df["ticker"] = ticker

        # Fetch financials from yfinance — free, no rate limits
        fin_df = fetch_yfinance_financials(ticker)

        if fin_df.empty:
            for col in ["revenue_growth_qoq", "asset_growth_qoq",
                        "liability_ratio", "cash_flow_to_assets", "net_income"]:
                price_df[col] = 0.0
        else:
            fin_df   = fin_df.dropna(subset=["date"]).sort_values("date")
            fin_df["date"] = fin_df["date"].astype("datetime64[us]")
            price_df = price_df.sort_values("date")
            fin_cols = ["date", "revenue_growth_qoq", "asset_growth_qoq",
                        "liability_ratio", "cash_flow_to_assets", "net_income"]
            price_df = pd.merge_asof(
                price_df, fin_df[fin_cols], on="date", direction="backward",
            )
            price_df[["revenue_growth_qoq", "asset_growth_qoq",
                       "liability_ratio", "cash_flow_to_assets",
                       "net_income"]] = (
                price_df[["revenue_growth_qoq", "asset_growth_qoq",
                           "liability_ratio", "cash_flow_to_assets",
                           "net_income"]].fillna(0)
            )

        all_frames.append(price_df)
        # Slight delay to be respectful to Yahoo Finance servers
        time.sleep(0.5)

    if not all_frames:
        raise RuntimeError("No data collected from API.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  Total rows collected: {len(combined)}")
    return combined


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    # ── Price-based features ───────────────────────────────────────────────
    df["price_change_pct"] = (df["close"] - df["open"]) / df["open"]
    df["return_1d"]  = df.groupby("ticker")["close"].pct_change(1)
    df["return_5d"]  = df.groupby("ticker")["close"].pct_change(5)
    df["return_10d"] = df.groupby("ticker")["close"].pct_change(10)
    df["volatility_10d"] = (
        df.groupby("ticker")["close"].pct_change()
        .rolling(10).std().reset_index(0, drop=True)
    )

    # ── Volume features ────────────────────────────────────────────────────
    df["volume_avg_10d"] = (
        df.groupby("ticker")["volume"].rolling(10).mean().reset_index(0, drop=True)
    )
    # Normalized volume spike detector — works the same across all price levels
    df["volume_anomaly_ratio"] = (
        df["volume"] / (df["volume_avg_10d"] + 1)
    ).clip(0, 10)

    # ── RSI — Wilder's EWM, grouped per ticker to prevent cross-ticker bleed ──
    def _wilder_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
        delta    = closes.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs       = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = df.groupby("ticker")["close"].transform(_wilder_rsi)

    # ── MACD — grouped to prevent cross-ticker bleed ───────────────────────
    ema12 = df.groupby("ticker")["close"].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    ema26 = df.groupby("ticker")["close"].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    df["macd_line"]   = ema12 - ema26
    df["macd_signal"] = df.groupby("ticker")["macd_line"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # ── Earnings surprise — did net income beat the prior quarter? ─────────
    df["earnings_surprise"] = df.groupby("ticker")["net_income"].transform(
        lambda x: x.diff().apply(
            lambda v: 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
        )
    ).fillna(0.0)

    # ── Clip fundamentals to prevent outlier drift ─────────────────────────
    df["revenue_growth_qoq"] = df["revenue_growth_qoq"].clip(-2.0, 2.0)
    df["asset_growth_qoq"]   = df["asset_growth_qoq"].clip(-2.0, 2.0)
    df["liability_ratio"]    = df["liability_ratio"].clip(0.0, 5.0)
    df["cash_flow_to_assets"]= df["cash_flow_to_assets"].clip(-5.0, 5.0)

    # 14 features — must match app.py build_model_features exactly
    feature_cols = [
        "price_change_pct", "return_1d", "return_5d", "return_10d",
        "volatility_10d", "volume_avg_10d", "volume_anomaly_ratio",
        "revenue_growth_qoq", "asset_growth_qoq", "liability_ratio",
        "cash_flow_to_assets", "rsi_14", "macd_hist", "earnings_surprise",
    ]

    X = df[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def build_labels(frame: pd.DataFrame, horizon: int = 5) -> pd.Series:
    future_close   = frame.groupby("ticker")["close"].shift(-horizon)
    target_returns = (future_close - frame["close"]) / frame["close"]
    # 1% threshold — realistic target that gives the model enough
    # positive examples to learn from (was 2% which was too strict)
    labels = (target_returns > 0.01).astype("Int64")
    labels[target_returns.isna()] = pd.NA
    return labels


def train():
    frame = collect_bulk_api_data()
    print(f"Loaded {len(frame)} rows across {frame['ticker'].nunique()} tickers")

    y = build_labels(frame)
    X = build_features(frame)

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    # Per-ticker time-based split — every ticker appears in both train and
    # test, and test rows are always chronologically after train rows
    train_idx, test_idx = [], []
    for ticker, grp in frame.loc[valid].groupby("ticker"):
        n      = len(grp)
        cutoff = int(n * 0.8)
        train_idx.extend(grp.index[:cutoff].tolist())
        test_idx.extend(grp.index[cutoff:].tolist())

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

    # Correct for class imbalance
    pos   = int(y_train.sum())
    neg   = len(y_train) - pos
    scale = neg / pos if pos > 0 else 1.0

    model = LGBMClassifier(
        n_estimators=500,
        random_state=42,
        learning_rate=0.03,
        scale_pos_weight=scale,
        num_leaves=31,
        min_child_samples=30,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- ML Performance Report ---")
    print("Test Accuracy Score:", round(accuracy_score(y_test, y_pred), 4))
    print("Test ROC AUC Score:", round(roc_auc_score(y_test, y_proba), 4))
    print("Feature count:", X_train.shape[1])

    # Timestamped backup so every retrain is recoverable
    if MODEL_PATH.exists():
        ts     = datetime.now().strftime("%Y%m%d_%H%M")
        backup = MODEL_PATH.with_name(f"lightgbm_stock_signal_{ts}.pkl")
        shutil.copy2(MODEL_PATH, backup)
        print(f"Backed up previous model to: {backup}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
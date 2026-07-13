from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from data_sources import fetch_yfinance_financials, fetch_yfinance_price_history

for ticker in ["AAPL", "MSFT"]:
    print(f"\n{'='*50}")
    print(f"TICKER: {ticker}")
    print(f"{'='*50}")

    fin_df = fetch_yfinance_financials(ticker)
    print(f"fin_df empty: {fin_df.empty}")
    if not fin_df.empty:
        print(f"fin_df shape: {fin_df.shape}")
        print(f"fin_df columns: {list(fin_df.columns)}")
        print(fin_df[["date","revenue","assets","liabilities",
                       "revenue_growth_qoq","asset_growth_qoq",
                       "liability_ratio","net_income"]].to_string())

    print()
    results = fetch_yfinance_price_history(ticker, days=30)
    print(f"Price rows fetched: {len(results)}")
    if results:
        price_df = pd.DataFrame(results).rename(columns={
            "t":"date","o":"open","h":"high","l":"low","c":"close","v":"volume"
        })
        price_df["date"] = pd.to_datetime(price_df["date"], unit="ms").dt.normalize().astype("datetime64[us]")
        price_df["ticker"] = ticker
        price_df = price_df.sort_values("date")

        if not fin_df.empty:
            fin_df2 = fin_df.dropna(subset=["date"]).sort_values("date").copy()
            fin_df2["date"] = fin_df2["date"].astype("datetime64[us]")
            fin_cols = ["date","revenue_growth_qoq","asset_growth_qoq",
                        "liability_ratio","cash_flow_to_assets","net_income"]
            merged = pd.merge_asof(price_df, fin_df2[fin_cols],
                                   on="date", direction="backward")
            print(f"\nAfter merge_asof — financial columns sample:")
            print(merged[["date","revenue_growth_qoq","asset_growth_qoq",
                          "liability_ratio","net_income"]].tail(5).to_string())
            non_zero = (merged["revenue_growth_qoq"] != 0).sum()
            print(f"\nNon-zero revenue_growth_qoq rows: {non_zero}/{len(merged)}")
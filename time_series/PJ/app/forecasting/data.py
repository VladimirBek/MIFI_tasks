from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_prices(ticker: str, years: int = 2) -> pd.Series:
    df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError("No data returned. Check ticker or Yahoo Finance availability.")
    if "Close" not in df.columns:
        raise ValueError("Unexpected data format from Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close" in df.columns.get_level_values(0)):
            close_df = df["Close"]
            # если есть колонка с тикером — берём её, иначе первую
            if ticker in close_df.columns:
                s = close_df[ticker]
            else:
                s = close_df.iloc[:, 0]
        else:
            raise ValueError("Unexpected data format from Yahoo Finance (MultiIndex without Close).")
    else:
        if "Close" not in df.columns:
            raise ValueError("Unexpected data format from Yahoo Finance.")
        s = df["Close"]

    # Гарантируем, что это Series
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    s = s.dropna().astype("float64")
    if len(s) < 120:
        raise ValueError("Not enough data to train models (need at least ~120 points).")
    s.index = pd.to_datetime(s.index)
    s.name = "close"
    return s

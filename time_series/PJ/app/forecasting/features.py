from __future__ import annotations

import pandas as pd

def build_lag_features(series: pd.Series, lags=(1,2,3,5,10,20), roll_windows=(5,20)) -> pd.DataFrame:
    """Notebook-style lag features + calendar features. Target is column 'y'."""
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for w in roll_windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["y"].shift(1).rolling(w).std()

    idx = df.index
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_month_start"] = idx.is_month_start.astype(int)
    df["is_month_end"] = idx.is_month_end.astype(int)

    return df.dropna()

def make_future_dates(last_date: pd.Timestamp, steps: int = 30) -> pd.DatetimeIndex:
    # Stocks trade on business days; approximate by business-day calendar.
    return pd.bdate_range(last_date + pd.Timedelta(days=1), periods=steps)

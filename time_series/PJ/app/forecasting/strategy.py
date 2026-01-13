from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


def buy_sell_points(forecast: pd.Series, order: int = 2):
    y = forecast.values
    if len(y) < (order * 2 + 1):
        return [], []

    mins = argrelextrema(y, np.less_equal, order=order)[0]
    maxs = argrelextrema(y, np.greater_equal, order=order)[0]

    mins = [i for i in mins if 0 < i < len(y) - 1]
    maxs = [i for i in maxs if 0 < i < len(y) - 1]

    buy_idx, sell_idx = [], []
    maxs_sorted = sorted(maxs)

    for mi in sorted(mins):
        candidates = [mx for mx in maxs_sorted if mx > mi]
        if not candidates:
            continue
        mx = candidates[0]
        maxs_sorted = [m for m in maxs_sorted if m > mx]
        buy_idx.append(mi)
        sell_idx.append(mx)

    buy_dates = [forecast.index[i] for i in buy_idx]
    sell_dates = [forecast.index[i] for i in sell_idx]
    return buy_dates, sell_dates


def simulate_profit(forecast: pd.Series, amount: float, buy_dates, sell_dates):
    cash = float(amount)
    shares = 0

    for b, s in zip(buy_dates, sell_dates):
        buy_price = float(forecast.loc[b])
        sell_price = float(forecast.loc[s])
        if buy_price <= 0:
            continue
        if sell_price <= buy_price:
            continue

        to_buy = int(cash // buy_price)
        if to_buy <= 0:
            continue

        cash -= to_buy * buy_price
        shares += to_buy

        cash += shares * sell_price
        shares = 0

    return cash - float(amount)

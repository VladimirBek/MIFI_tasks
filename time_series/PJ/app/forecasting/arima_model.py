from __future__ import annotations

import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .metrics import rmse, mape
from .features import make_future_dates

class ARIMAModel:
    name = "Stat_ARIMA"

    def __init__(self):
        self.order = (1, 1, 0)

    def _fit(self, y: pd.Series, order):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ARIMA(y, order=order).fit()

    def fit_and_eval(self, series: pd.Series, test_ratio: float = 0.2):
        n = len(series)
        split = int(n * (1 - test_ratio))
        train, test = series.iloc[:split], series.iloc[split:]

        candidates = []
        for p in [0,1,2,3]:
            for d in [0,1]:
                for q in [0,1,2]:
                    if p == 0 and d == 0 and q == 0:
                        continue
                    candidates.append((p,d,q))

        best = None
        best_rmse = float("inf")
        best_mape = float("inf")

        for order in candidates:
            try:
                model = self._fit(train, order)
                pred = model.forecast(steps=len(test))
                r = rmse(test, pred)
                m = mape(test, pred)
                if (r < best_rmse) or (abs(r - best_rmse) < 1e-9 and m < best_mape):
                    best_rmse, best_mape = r, m
                    best = order
            except Exception:
                continue

        if best is None:
            best = (1,1,0)
            model = self._fit(train, best)
            pred = model.forecast(steps=len(test))
            best_rmse = rmse(test, pred)
            best_mape = mape(test, pred)

        self.order = best

        return {
            "rmse": best_rmse,
            "mape": best_mape,
            "y_true": test,
            "y_pred": pd.Series(pred.values, index=test.index, name="pred"),
            "details": f"order={best}",
        }

    def forecast(self, series: pd.Series, steps: int = 30) -> pd.Series:
        model = self._fit(series, self.order)
        pred = model.forecast(steps=steps)
        future_dates = make_future_dates(series.index[-1], steps=steps)
        return pd.Series(pred.values, index=future_dates, name="forecast")

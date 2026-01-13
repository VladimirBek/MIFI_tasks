from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .features import build_lag_features, make_future_dates
from .metrics import rmse, mape

class RidgeLagModel:
    name = "ML_Ridge_Lags"

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=0)),
        ])
        self.feature_cols: list[str] = []

    def fit_and_eval(self, series: pd.Series, test_ratio: float = 0.2):
        df = build_lag_features(series)
        self.feature_cols = [c for c in df.columns if c != "y"]

        split = int(len(df) * (1 - test_ratio))
        train, test = df.iloc[:split], df.iloc[split:]

        X_train, y_train = train[self.feature_cols], train["y"]
        X_test, y_test = test[self.feature_cols], test["y"]

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        return {
            "rmse": rmse(y_test, y_pred),
            "mape": mape(y_test, y_pred),
            "y_true": y_test,
            "y_pred": pd.Series(y_pred, index=y_test.index, name="pred"),
        }

    def forecast(self, series: pd.Series, steps: int = 30) -> pd.Series:
        df = build_lag_features(series)
        self.feature_cols = [c for c in df.columns if c != "y"]
        self.model.fit(df[self.feature_cols], df["y"])

        hist = series.copy()
        future_dates = make_future_dates(hist.index[-1], steps=steps)
        preds = []

        for d in future_dates:
            tmp = pd.concat([hist, pd.Series([np.nan], index=[d], name=hist.name)])
            feat_df = build_lag_features(tmp)
            x = feat_df.loc[[feat_df.index[-1]], self.feature_cols]
            yhat = float(self.model.predict(x)[0])
            preds.append(yhat)
            hist.loc[d] = yhat

        return pd.Series(preds, index=future_dates, name="forecast")

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass

from .ml_model import RidgeLagModel
from .arima_model import ARIMAModel
from .lstm_model import LSTMModel


@dataclass
class ModelResult:
    name: str
    rmse: float
    mape: float
    details: str = ""


def select_best_model(series: pd.Series):
    models = [
        RidgeLagModel(),
        ARIMAModel(),
        LSTMModel(window=30, epochs=15, batch_size=32),
    ]

    results: list[ModelResult] = []
    for m in models:
        blob = m.fit_and_eval(series, test_ratio=0.2)
        results.append(ModelResult(
            name=getattr(m, "name", m.__class__.__name__),
            rmse=float(blob["rmse"]),
            mape=float(blob["mape"]),
            details=str(blob.get("details", "")),
        ))

    results_sorted = sorted(results, key=lambda r: (r.rmse, r.mape))
    best_name = results_sorted[0].name

    best_model = next(m for m in models if getattr(m, "name", m.__class__.__name__) == best_name)
    return best_model, results_sorted

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd


def plot_history_and_forecast(history: pd.Series, forecast: pd.Series, title: str) -> io.BytesIO:
    buf = io.BytesIO()
    plt.figure(figsize=(12, 6))
    hist_to_plot = history.tail(180)
    plt.plot(hist_to_plot.index, hist_to_plot.values, label="Исторические данные")
    plt.plot(forecast.index, forecast.values, label="Прогноз")
    plt.title(title)
    plt.xlabel("Дата")
    plt.ylabel("Цена")
    plt.legend()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=160)
    plt.close()
    buf.seek(0)
    return buf

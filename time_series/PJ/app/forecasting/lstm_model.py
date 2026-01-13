from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers

from .metrics import rmse, mape
from .features import make_future_dates

class LSTMModel:
    name = "NN_LSTM"

    def __init__(self, window: int = 30, epochs: int = 15, batch_size: int = 32):
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model: tf.keras.Model | None = None

        # Reduce TF thread contention in small containers
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass

    def _make_xy(self, arr: np.ndarray):
        X, y = [], []
        for i in range(self.window, len(arr)):
            X.append(arr[i-self.window:i, 0])
            y.append(arr[i, 0])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return X[..., None], y

    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=(self.window, 1)),
            layers.LSTM(32),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit_and_eval(self, series: pd.Series, test_ratio: float = 0.2):
        values = series.values.reshape(-1, 1).astype("float32")
        scaled = self.scaler.fit_transform(values)

        split = int(len(scaled) * (1 - test_ratio))
        train_scaled = scaled[:split]
        test_scaled = scaled[split - self.window:]  # keep window context

        X_train, y_train = self._make_xy(train_scaled)
        X_test, y_test = self._make_xy(test_scaled)

        self.model = self._build_model()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ]
        self.model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=callbacks,
        )

        y_pred_scaled = self.model.predict(X_test, verbose=0).reshape(-1, 1)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).ravel()

        y_test_true = self.scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        test_index = series.index[split:][:len(y_pred)]

        return {
            "rmse": rmse(y_test_true, y_pred),
            "mape": mape(y_test_true, y_pred),
            "y_true": pd.Series(y_test_true, index=test_index, name="true"),
            "y_pred": pd.Series(y_pred, index=test_index, name="pred"),
        }

    def forecast(self, series: pd.Series, steps: int = 30) -> pd.Series:
        # Train quickly if needed
        if self.model is None:
            self.fit_and_eval(series, test_ratio=0.2)

        values = series.values.reshape(-1, 1).astype("float32")
        scaled = self.scaler.fit_transform(values)

        window_seq = scaled[-self.window:].reshape(1, self.window, 1).astype("float32")
        preds = []
        for _ in range(steps):
            p = float(self.model.predict(window_seq, verbose=0)[0, 0])
            preds.append(p)
            next_seq = np.append(window_seq[0, 1:, 0], p).reshape(1, self.window, 1)
            window_seq = next_seq.astype("float32")

        preds = np.array(preds).reshape(-1, 1)
        preds_inv = self.scaler.inverse_transform(preds).ravel()
        future_dates = make_future_dates(series.index[-1], steps=steps)
        return pd.Series(preds_inv, index=future_dates, name="forecast")

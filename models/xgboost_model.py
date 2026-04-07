"""XGBoost forecaster with hand-crafted technical features."""

from __future__ import annotations

import gc
import time

import numpy as np
import xgboost as xgb


def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.convolve(gains, np.ones(period) / period, mode="full")[:len(prices)]
    avg_loss = np.convolve(losses, np.ones(period) / period, mode="full")[:len(prices)]
    avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def _ema(arr, span):
        alpha = 2.0 / (span + 1)
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    ema12 = _ema(prices, 12)
    ema26 = _ema(prices, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    return macd_line, signal


def _build_features(prices: np.ndarray) -> np.ndarray:
    """Build feature matrix from price series. Returns (n_valid, n_features)."""
    n = len(prices)
    lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    windows = [5, 10, 20, 60]
    max_lookback = max(max(lags), max(windows))

    rsi = _rsi(prices)
    macd_line, macd_signal = _macd(prices)

    rows = []
    for i in range(max_lookback, n):
        feat = []
        for lag in lags:
            feat.append(prices[i - lag])
        for w in windows:
            window = prices[i - w : i]
            feat.append(np.mean(window))
            feat.append(np.std(window))
        feat.append(rsi[i])
        feat.append(macd_line[i])
        feat.append(macd_signal[i])
        rows.append(feat)

    return np.array(rows, dtype=np.float32), max_lookback


class XGBoostForecaster:
    name = "XGBoost"

    def __init__(self):
        self._params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    def predict(self, history: np.ndarray, horizon: int) -> dict:
        prices = history.astype(np.float64)
        features, offset = _build_features(prices)

        # Targets: next-day price
        targets = prices[offset + 1 : offset + 1 + len(features)]
        # Trim features to match targets
        X = features[: len(targets)]
        y = targets

        t0 = time.perf_counter()

        # Train point model
        model = xgb.XGBRegressor(**self._params)
        model.fit(X, y, verbose=False)

        # Train quantile models
        model_q10 = xgb.XGBRegressor(
            **{**self._params, "objective": "reg:quantileerror", "quantile_alpha": 0.1}
        )
        model_q90 = xgb.XGBRegressor(
            **{**self._params, "objective": "reg:quantileerror", "quantile_alpha": 0.9}
        )
        model_q10.fit(X, y, verbose=False)
        model_q90.fit(X, y, verbose=False)

        # Recursive multi-step forecast
        current_prices = prices.copy()
        point_preds = []
        q10_preds = []
        q90_preds = []

        for _ in range(horizon):
            feat, off = _build_features(current_prices)
            last_feat = feat[-1:].reshape(1, -1)
            p = model.predict(last_feat)[0]
            p10 = model_q10.predict(last_feat)[0]
            p90 = model_q90.predict(last_feat)[0]
            point_preds.append(p)
            q10_preds.append(p10)
            q90_preds.append(p90)
            current_prices = np.append(current_prices, p)

        elapsed = time.perf_counter() - t0

        return {
            "point_forecast": np.array(point_preds),
            "quantile_10": np.array(q10_preds),
            "quantile_90": np.array(q90_preds),
            "inference_time_seconds": elapsed,
        }

    def cleanup(self):
        gc.collect()

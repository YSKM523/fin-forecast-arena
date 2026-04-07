"""TimesFM 2.5 forecaster wrapper."""

from __future__ import annotations

import gc
import time

import numpy as np


class TimesFMForecaster:
    name = "TimesFM-2.5"

    def __init__(self):
        from huggingface_hub import hf_hub_download
        from timesfm import ForecastConfig, TimesFM_2p5_200M_torch

        weights = hf_hub_download(
            "google/timesfm-2.5-200m-pytorch", "model.safetensors"
        )
        self._model = TimesFM_2p5_200M_torch(torch_compile=False)
        self._model.model.load_checkpoint(weights, torch_compile=False)

        self._fc = ForecastConfig(max_context=512, max_horizon=128)
        self._model.compile(self._fc)

    def predict(self, history: np.ndarray, horizon: int) -> dict:
        arr = history.astype(np.float32)

        t0 = time.perf_counter()
        points, quantiles = self._model.forecast(horizon, [arr])
        elapsed = time.perf_counter() - t0

        point = points[0, :horizon]
        # quantiles shape: (1, horizon, 10) — columns map to
        # [q10, q20, q30, q40, q50(point), q60, q70, q80, q90, extra]
        q10 = quantiles[0, :horizon, 0]
        q90 = quantiles[0, :horizon, 8]

        return {
            "point_forecast": point,
            "quantile_10": q10,
            "quantile_90": q90,
            "inference_time_seconds": elapsed,
        }

    def cleanup(self):
        del self._model
        gc.collect()

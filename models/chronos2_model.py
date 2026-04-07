"""Chronos-2 forecaster wrapper."""

from __future__ import annotations

import gc
import time

import numpy as np
import torch


class Chronos2Forecaster:
    name = "Chronos-2"

    def __init__(self):
        from chronos import Chronos2Pipeline

        self._pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map="cpu",
            dtype=torch.float32,
        )

    def predict(self, history: np.ndarray, horizon: int) -> dict:
        ctx = torch.tensor(history, dtype=torch.float32).reshape(1, 1, -1)

        t0 = time.perf_counter()
        # Returns list of tensors; each shape (batch, num_samples, horizon)
        forecast = self._pipeline.predict(ctx, prediction_length=horizon)
        elapsed = time.perf_counter() - t0

        samples = forecast[0].numpy().squeeze(0)  # (num_samples, horizon)
        point = np.median(samples, axis=0)
        q10 = np.percentile(samples, 10, axis=0)
        q90 = np.percentile(samples, 90, axis=0)

        return {
            "point_forecast": point,
            "quantile_10": q10,
            "quantile_90": q90,
            "inference_time_seconds": elapsed,
        }

    def cleanup(self):
        del self._pipeline
        gc.collect()

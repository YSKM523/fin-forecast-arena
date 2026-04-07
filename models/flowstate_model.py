"""FlowState (IBM Granite) forecaster wrapper."""

from __future__ import annotations

import gc
import time

import numpy as np
import torch


class FlowStateForecaster:
    name = "FlowState"

    def __init__(self):
        from tsfm_public import FlowStateForPrediction

        self._model = FlowStateForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-flowstate-r1",
        ).to("cpu")

    def predict(self, history: np.ndarray, horizon: int) -> dict:
        # FlowState: (context_length, batch=1, channels=1), batch_first=False
        ts = torch.tensor(history, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

        t0 = time.perf_counter()
        out = self._model(
            ts,
            scale_factor=1.0,
            prediction_length=horizon,
            batch_first=False,
        )
        elapsed = time.perf_counter() - t0

        # prediction_outputs: (batch, horizon, channels)
        point = out.prediction_outputs[0, :, 0].detach().numpy()

        # quantile_outputs: (batch, 9_quantiles, horizon, channels)
        # quantiles are [0.1, 0.2, ..., 0.9]
        q_all = out.quantile_outputs[0, :, :, 0].detach().numpy()  # (9, horizon)
        q10 = q_all[0]  # index 0 = 0.1
        q90 = q_all[8]  # index 8 = 0.9

        return {
            "point_forecast": point,
            "quantile_10": q10,
            "quantile_90": q90,
            "inference_time_seconds": elapsed,
        }

    def cleanup(self):
        del self._model
        gc.collect()

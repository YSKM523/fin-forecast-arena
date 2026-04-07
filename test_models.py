"""Smoke-test each forecasting model one at a time on a sine wave."""

import gc
import sys
import numpy as np
import torch

HORIZON = 12
context = np.sin(np.linspace(0, 4 * np.pi, 120)).astype(np.float32)


def banner(name):
    print(f"\n{'='*60}\n  Testing: {name}\n{'='*60}", flush=True)


def cleanup():
    gc.collect()


# ── 1. Chronos-2 ────────────────────────────────────────────
banner("Chronos-2 (amazon/chronos-2)")
try:
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        dtype=torch.float32,
    )
    # Chronos-2 expects (n_series, n_variates, history_length)
    ctx_tensor = torch.tensor(context).reshape(1, 1, -1)
    forecast = pipeline.predict(ctx_tensor, prediction_length=HORIZON)

    # forecast is list of tensors with shape (batch, num_samples, horizon)
    arr = forecast[0].numpy()
    print(f"  shape : {arr.shape}")
    median = np.median(arr, axis=-2)  # median over samples
    print(f"  first3: {median.flatten()[:3]}")
    print("✓ Chronos-2 OK")
    del pipeline, forecast, ctx_tensor
    cleanup()
except Exception as e:
    print(f"✗ Chronos-2 FAILED: {e}")
    cleanup()

# ── 2. TimesFM 2.5 ──────────────────────────────────────────
banner("TimesFM 2.5 (google/timesfm-2.5-200m-pytorch)")
try:
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(
        "google/timesfm-2.5-200m-pytorch", "model.safetensors"
    )
    tfm = TimesFM_2p5_200M_torch(torch_compile=False)
    tfm.model.load_checkpoint(weights, torch_compile=False)

    fc = ForecastConfig(max_context=128, max_horizon=128)
    tfm.compile(fc)

    points, quantiles = tfm.forecast(HORIZON, [context])
    print(f"  shape : {points.shape}")
    print(f"  first3: {points[0, :3]}")
    print("✓ TimesFM 2.5 OK")
    del tfm, points, quantiles
    cleanup()
except Exception as e:
    print(f"✗ TimesFM 2.5 FAILED: {e}")
    cleanup()

# ── 3. FlowState → fallback Moirai 2.0 ──────────────────────
banner("FlowState (ibm-granite/granite-timeseries-flowstate-r1)")
flowstate_ok = False
try:
    from tsfm_public import FlowStateForPrediction

    predictor = FlowStateForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-flowstate-r1",
    ).to("cpu")

    # FlowState expects (context_length, batch_size, n_channels) with batch_first=False
    ts = torch.tensor(context).unsqueeze(-1).unsqueeze(-1)  # (120, 1, 1)
    forecast = predictor(
        ts,
        scale_factor=1.0,
        prediction_length=HORIZON,
        batch_first=False,
    )

    arr = forecast.prediction_outputs.detach().numpy()
    print(f"  shape : {arr.shape}")
    vals = arr.flatten()[:HORIZON]
    print(f"  first3: {vals[:3]}")
    print("✓ FlowState OK")
    flowstate_ok = True
    del predictor, forecast, ts
    cleanup()
except Exception as e:
    print(f"✗ FlowState FAILED: {e}")
    cleanup()

if not flowstate_ok:
    banner("Moirai 2.0 fallback (Salesforce/moirai-2.0-R-small)")
    print("  ⚠ Moirai 2.0 requires uni2ts which needs torch<2.5")
    print("  ⚠ Current torch is incompatible — skipping")
    print("✗ Moirai 2.0 SKIPPED: torch version incompatible with uni2ts")

print(f"\n{'='*60}\n  All tests complete.\n{'='*60}")

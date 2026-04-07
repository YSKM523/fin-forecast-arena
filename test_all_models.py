"""Test all 4 model wrappers on NVDA close prices, one at a time."""

import gc
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

# Load NVDA close prices — last 252 trading days
df = pd.read_parquet("data/cache/NVDA.parquet")
close = df["Close"].values.astype(np.float64)[-252:]
print(f"NVDA close: {len(close)} days, range ${close.min():.2f}–${close.max():.2f}\n")

HORIZON = 5

MODEL_CLASSES = [
    ("models.chronos2_model", "Chronos2Forecaster"),
    ("models.timesfm_model", "TimesFMForecaster"),
    ("models.flowstate_model", "FlowStateForecaster"),
    ("models.xgboost_model", "XGBoostForecaster"),
]

for module_path, class_name in MODEL_CLASSES:
    print(f"{'='*60}")
    mod = __import__(module_path, fromlist=[class_name])
    cls = getattr(mod, class_name)

    try:
        print(f"  Loading {cls.name}...")
        model = cls()
        result = model.predict(close, HORIZON)

        pf = result["point_forecast"]
        print(f"  ✓ {cls.name}")
        print(f"    point_forecast : {pf}")
        if result.get("quantile_10") is not None:
            print(f"    quantile_10    : {result['quantile_10']}")
            print(f"    quantile_90    : {result['quantile_90']}")
        print(f"    inference_time : {result['inference_time_seconds']:.3f}s")

        model.cleanup()
        del model, result
    except Exception as e:
        print(f"  ✗ {cls.name} FAILED: {e}")

    gc.collect()
    print()

print("All models tested.")

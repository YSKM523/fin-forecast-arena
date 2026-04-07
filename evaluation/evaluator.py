"""Rolling backtest evaluator.

Outer loop: models (load → evaluate all tickers → cleanup → gc).
Inner loop: tickers × rolling windows.

For each of the past 60 trading days, uses the prior 252 days as context,
predicts the next 5 days, and compares with actuals.

After the 3 foundation models run, computes 3 ensemble methods and evaluates
them on the same windows.
"""

from __future__ import annotations

import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import compute_window_metrics, sharpe_ratio
from pipeline.ensemble import (
    compute_trailing_maes,
    dynamic_weighted,
    equal_weight,
    majority_vote_direction,
)

# ── Config ───────────────────────────────────────────────────
CONTEXT_LEN = 252
HORIZON = 5
N_WINDOWS = 60

TICKERS = [
    "TSM", "NVDA", "AMD", "INTC", "ASML", "AVGO", "QCOM",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "SPY", "QQQ", "SOXX",
]

FOUNDATION_SPECS = [
    ("models.chronos2_model", "Chronos2Forecaster"),
    ("models.timesfm_model", "TimesFMForecaster"),
    ("models.flowstate_model", "FlowStateForecaster"),
]

BASELINE_SPECS = [
    ("models.xgboost_model", "XGBoostForecaster"),
]

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "evaluations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_close(ticker: str) -> np.ndarray:
    df = pd.read_parquet(CACHE_DIR / f"{ticker}.parquet")
    return df["Close"].values.astype(np.float64)


def _origin_range(n: int):
    """Return (first_origin, last_origin) indices for the rolling windows."""
    last_origin = n - 1 - HORIZON
    first_origin = last_origin - N_WINDOWS + 1
    return first_origin, last_origin


def _evaluate_ticker(
    predictions: list[np.ndarray],
    actuals_list: list[np.ndarray],
    contexts_list: list[np.ndarray],
    origins_list: list[float],
) -> dict:
    """Compute aggregated metrics for one (model, ticker) pair."""
    window_metrics = []
    signals_1d = []
    returns_1d = []

    for pf, act, ctx, origin_price in zip(
        predictions, actuals_list, contexts_list, origins_list
    ):
        wm = compute_window_metrics(pf, act, ctx, origin_price)
        window_metrics.append(wm)

        signal = 1.0 if pf[0] > origin_price else -1.0
        ret = (act[0] - origin_price) / origin_price
        signals_1d.append(signal)
        returns_1d.append(ret)

    avg = {k: np.nanmean([w[k] for w in window_metrics]) for k in window_metrics[0]}
    sr = sharpe_ratio(np.array(signals_1d), np.array(returns_1d))
    return {
        "mae": round(avg["mae"], 4),
        "rmse": round(avg["rmse"], 4),
        "mase": round(avg["mase"], 4),
        "dir_acc": round(avg["dir_acc"], 4),
        "sharpe": round(sr, 4),
    }


def run_backtest():
    # Pre-load all close price arrays
    all_close = {t: load_close(t) for t in TICKERS}

    min_needed = CONTEXT_LEN + N_WINDOWS + HORIZON
    for t, c in all_close.items():
        assert len(c) >= min_needed, f"{t}: only {len(c)} days, need {min_needed}"

    # ── Pre-compute actuals, contexts, origin prices per ticker ──
    # These are the same for every model.
    ticker_actuals: dict[str, list[np.ndarray]] = {}
    ticker_contexts: dict[str, list[np.ndarray]] = {}
    ticker_origins: dict[str, list[float]] = {}

    for ticker in TICKERS:
        close = all_close[ticker]
        first_origin, last_origin = _origin_range(len(close))
        acts, ctxs, origs = [], [], []
        for origin in range(first_origin, last_origin + 1):
            ctx_start = origin - CONTEXT_LEN + 1
            ctxs.append(close[ctx_start : origin + 1])
            acts.append(close[origin + 1 : origin + 1 + HORIZON])
            origs.append(close[origin])
        ticker_actuals[ticker] = acts
        ticker_contexts[ticker] = ctxs
        ticker_origins[ticker] = origs

    rows = []

    # ── Storage for foundation model predictions ─────────────
    # fm_preds[ticker][model_name][window_idx] = np.ndarray (HORIZON,)
    fm_preds: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    fm_names: list[str] = []

    # ── Phase 1: Run foundation models (one at a time) ───────
    for mod_path, cls_name in FOUNDATION_SPECS:
        print(f"\n{'='*70}")
        print(f"  MODEL: {cls_name}")
        print(f"{'='*70}", flush=True)
        mod = __import__(mod_path, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        model = cls()
        model_name = cls.name
        fm_names.append(model_name)

        model_t0 = time.perf_counter()

        for ticker in TICKERS:
            contexts = ticker_contexts[ticker]
            preds_for_ticker = []
            for w_idx in range(N_WINDOWS):
                result = model.predict(contexts[w_idx], HORIZON)
                preds_for_ticker.append(result["point_forecast"][:HORIZON].copy())
            fm_preds[ticker][model_name] = preds_for_ticker

            # Evaluate this single model on this ticker
            metrics = _evaluate_ticker(
                preds_for_ticker,
                ticker_actuals[ticker],
                ticker_contexts[ticker],
                ticker_origins[ticker],
            )
            rows.append({"model": model_name, "ticker": ticker, **metrics})
            print(f"  {ticker:5s}  MAE={metrics['mae']:7.2f}  RMSE={metrics['rmse']:7.2f}  "
                  f"MASE={metrics['mase']:.3f}  DirAcc={metrics['dir_acc']:.2%}  "
                  f"Sharpe={metrics['sharpe']:+.2f}", flush=True)

        elapsed = time.perf_counter() - model_t0
        print(f"  [{model_name} total: {elapsed:.1f}s]", flush=True)

        model.cleanup()
        del model
        gc.collect()

    # ── Phase 2: Run baseline models (XGBoost) ───────────────
    for mod_path, cls_name in BASELINE_SPECS:
        print(f"\n{'='*70}")
        print(f"  MODEL: {cls_name}")
        print(f"{'='*70}", flush=True)
        mod = __import__(mod_path, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        model = cls()
        model_name = cls.name

        model_t0 = time.perf_counter()

        for ticker in TICKERS:
            contexts = ticker_contexts[ticker]
            preds_for_ticker = []
            for w_idx in range(N_WINDOWS):
                result = model.predict(contexts[w_idx], HORIZON)
                preds_for_ticker.append(result["point_forecast"][:HORIZON].copy())

            metrics = _evaluate_ticker(
                preds_for_ticker,
                ticker_actuals[ticker],
                ticker_contexts[ticker],
                ticker_origins[ticker],
            )
            rows.append({"model": model_name, "ticker": ticker, **metrics})
            print(f"  {ticker:5s}  MAE={metrics['mae']:7.2f}  RMSE={metrics['rmse']:7.2f}  "
                  f"MASE={metrics['mase']:.3f}  DirAcc={metrics['dir_acc']:.2%}  "
                  f"Sharpe={metrics['sharpe']:+.2f}", flush=True)

        elapsed = time.perf_counter() - model_t0
        print(f"  [{model_name} total: {elapsed:.1f}s]", flush=True)

        model.cleanup()
        del model
        gc.collect()

    # ── Phase 3: Compute ensemble forecasts from stored predictions ──
    ensemble_methods = {
        "Ens-EqualWt": "equal",
        "Ens-DynWt": "dynamic",
        "Ens-MajVote": "majority",
    }

    for ens_name, method in ensemble_methods.items():
        print(f"\n{'='*70}")
        print(f"  ENSEMBLE: {ens_name}")
        print(f"{'='*70}", flush=True)

        for ticker in TICKERS:
            actuals = ticker_actuals[ticker]
            contexts = ticker_contexts[ticker]
            origins = ticker_origins[ticker]
            model_preds = fm_preds[ticker]  # {model_name: list[np.ndarray]}

            ens_forecasts = []
            for w_idx in range(N_WINDOWS):
                forecasts = [model_preds[m][w_idx] for m in fm_names]
                origin_price = origins[w_idx]

                if method == "equal":
                    ens_forecasts.append(equal_weight(forecasts))

                elif method == "dynamic":
                    trailing = compute_trailing_maes(
                        model_preds, actuals, w_idx,
                        lookback=20, model_names=fm_names,
                    )
                    recent_maes = [trailing[m] for m in fm_names]
                    ens_forecasts.append(dynamic_weighted(forecasts, recent_maes))

                elif method == "majority":
                    ens_forecasts.append(
                        majority_vote_direction(forecasts, origin_price)
                    )

            metrics = _evaluate_ticker(ens_forecasts, actuals, contexts, origins)
            rows.append({"model": ens_name, "ticker": ticker, **metrics})
            print(f"  {ticker:5s}  MAE={metrics['mae']:7.2f}  RMSE={metrics['rmse']:7.2f}  "
                  f"MASE={metrics['mase']:.3f}  DirAcc={metrics['dir_acc']:.2%}  "
                  f"Sharpe={metrics['sharpe']:+.2f}", flush=True)

    # ── Save & summarise ─────────────────────────────────────
    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "backtest_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}\n")

    # Summary: average across all tickers per model/ensemble
    print("=" * 70)
    print("  SUMMARY  (mean across 15 tickers)")
    print("=" * 70)
    summary = df.groupby("model")[["mae", "rmse", "mase", "dir_acc", "sharpe"]].mean()
    summary = summary.round(4)

    # Sort: single models first, then ensembles
    order = fm_names + [cls_name_to_name(c) for _, c in BASELINE_SPECS] + list(ensemble_methods.keys())
    order = [m for m in order if m in summary.index]
    summary = summary.loc[order]

    print(summary.to_string())

    # ── Best-of comparison ───────────────────────────────────
    print(f"\n{'─'*70}")
    single_models = fm_names + [cls_name_to_name(c) for _, c in BASELINE_SPECS]
    single = summary.loc[summary.index.isin(single_models)]
    ensembles = summary.loc[summary.index.isin(ensemble_methods.keys())]

    best_single_mase = single["mase"].min()
    best_single_name = single["mase"].idxmin()
    best_ens_mase = ensembles["mase"].min()
    best_ens_name = ensembles["mase"].idxmin()

    print(f"  Best single model : {best_single_name:15s}  MASE={best_single_mase:.4f}")
    print(f"  Best ensemble     : {best_ens_name:15s}  MASE={best_ens_mase:.4f}")
    if best_ens_mase < best_single_mase:
        pct = (best_single_mase - best_ens_mase) / best_single_mase * 100
        print(f"  → Ensemble beats best single model by {pct:.1f}% on MASE")
    else:
        print(f"  → Best single model wins on MASE")

    best_single_da = single["dir_acc"].max()
    best_single_da_name = single["dir_acc"].idxmax()
    best_ens_da = ensembles["dir_acc"].max()
    best_ens_da_name = ensembles["dir_acc"].idxmax()
    print(f"\n  Best single DirAcc: {best_single_da_name:15s}  DirAcc={best_single_da:.4f}")
    print(f"  Best ens DirAcc   : {best_ens_da_name:15s}  DirAcc={best_ens_da:.4f}")
    if best_ens_da > best_single_da:
        print(f"  → Ensemble beats best single model on directional accuracy")
    else:
        print(f"  → Best single model wins on directional accuracy")

    print()
    return df


def cls_name_to_name(cls_name: str) -> str:
    """Map class names to display names."""
    mapping = {
        "Chronos2Forecaster": "Chronos-2",
        "TimesFMForecaster": "TimesFM-2.5",
        "FlowStateForecaster": "FlowState",
        "XGBoostForecaster": "XGBoost",
    }
    return mapping.get(cls_name, cls_name)


if __name__ == "__main__":
    run_backtest()

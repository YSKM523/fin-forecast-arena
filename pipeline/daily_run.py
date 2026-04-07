"""Daily pipeline: fetch data, predict, evaluate past predictions, plot.

Designed to run once per trading day after market close.
"""

from __future__ import annotations

import datetime as dt
import gc
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.fetcher import TICKERS, fetch_all, load_cached
from evaluation.metrics import compute_window_metrics, mae, sharpe_ratio
from pipeline.ensemble import (
    dynamic_weighted,
    equal_weight,
    majority_vote_direction,
)

# ── Paths ────────────────────────────────────────────────────
PRED_DIR = ROOT / "results" / "predictions"
EVAL_DIR = ROOT / "results" / "evaluations"
PLOT_DIR = ROOT / "results" / "plots"
for d in (PRED_DIR, EVAL_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

CUMULATIVE_CSV = EVAL_DIR / "cumulative.csv"

# ── Config ───────────────────────────────────────────────────
CONTEXT_LEN = 252
HORIZON = 5

FOUNDATION_SPECS = [
    ("models.chronos2_model", "Chronos2Forecaster"),
    ("models.timesfm_model", "TimesFMForecaster"),
    ("models.flowstate_model", "FlowStateForecaster"),
]
BASELINE_SPECS = [
    ("models.xgboost_model", "XGBoostForecaster"),
]
FM_NAMES = ["Chronos-2", "TimesFM-2.5", "FlowState"]
ENSEMBLE_NAMES = ["Ens-EqualWt", "Ens-DynWt", "Ens-MajVote"]
ALL_MODEL_NAMES = FM_NAMES + ["XGBoost"] + ENSEMBLE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  Step 1 — Fetch latest price data
# ══════════════════════════════════════════════════════════════
def step_fetch() -> dict[str, pd.DataFrame]:
    log.info("Step 1: Updating price data …")
    data = fetch_all()
    for t, df in data.items():
        log.info("  %s  %s → %s  (%d rows)",
                 t, df.index.min().date(), df.index.max().date(), len(df))
    return data


# ══════════════════════════════════════════════════════════════
#  Step 2 — Run all models + ensembles, predict next 5 days
# ══════════════════════════════════════════════════════════════
def step_predict(today: str) -> pd.DataFrame:
    log.info("Step 2: Running predictions for %s …", today)

    rows = []
    # {ticker: {model_name: np.ndarray}} for ensemble computation
    fm_preds: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    # Prepare contexts
    contexts: dict[str, np.ndarray] = {}
    origin_prices: dict[str, float] = {}
    for ticker in TICKERS:
        df = load_cached(ticker)
        close = df["Close"].values.astype(np.float64)
        contexts[ticker] = close[-CONTEXT_LEN:]
        origin_prices[ticker] = float(close[-1])

    def _run_model_specs(specs, store_fm=False):
        for mod_path, cls_name in specs:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            log.info("  Loading %s …", cls.name)
            model = cls()
            model_name = cls.name

            for ticker in TICKERS:
                result = model.predict(contexts[ticker], HORIZON)
                pf = result["point_forecast"][:HORIZON]

                if store_fm:
                    fm_preds[ticker][model_name] = pf.copy()

                row = {
                    "prediction_date": today,
                    "ticker": ticker,
                    "model": model_name,
                    "origin_price": origin_prices[ticker],
                }
                for h in range(HORIZON):
                    row[f"h{h+1}"] = float(pf[h])
                rows.append(row)

            log.info("  %s done — cleaning up", model_name)
            model.cleanup()
            del model
            gc.collect()

    _run_model_specs(FOUNDATION_SPECS, store_fm=True)
    _run_model_specs(BASELINE_SPECS, store_fm=False)

    # ── Ensembles ────────────────────────────────────────────
    log.info("  Computing ensembles …")

    # For dynamic weights, use cumulative history if available
    cum_maes = _load_trailing_maes()

    for ticker in TICKERS:
        forecasts = [fm_preds[ticker][m] for m in FM_NAMES]
        origin_price = origin_prices[ticker]

        # Equal weight
        ew = equal_weight(forecasts)
        # Dynamic weight
        if cum_maes is not None:
            recent = [cum_maes.get(m, 1.0) for m in FM_NAMES]
        else:
            recent = [1.0, 1.0, 1.0]
        dw = dynamic_weighted(forecasts, recent)
        # Majority vote
        mv = majority_vote_direction(forecasts, origin_price)

        for ens_name, pf in [
            ("Ens-EqualWt", ew), ("Ens-DynWt", dw), ("Ens-MajVote", mv)
        ]:
            row = {
                "prediction_date": today,
                "ticker": ticker,
                "model": ens_name,
                "origin_price": origin_price,
            }
            for h in range(HORIZON):
                row[f"h{h+1}"] = float(pf[h])
            rows.append(row)

    pred_df = pd.DataFrame(rows)
    out_path = PRED_DIR / f"{today}.csv"
    pred_df.to_csv(out_path, index=False)
    log.info("  Predictions saved to %s  (%d rows)", out_path.name, len(pred_df))
    return pred_df


def _load_trailing_maes() -> dict[str, float] | None:
    """Load average MAE per foundation model from cumulative results."""
    if not CUMULATIVE_CSV.exists():
        return None
    cum = pd.read_csv(CUMULATIVE_CSV)
    if cum.empty:
        return None
    recent = cum[cum["model"].isin(FM_NAMES)]
    if recent.empty:
        return None
    return recent.groupby("model")["mae"].mean().to_dict()


# ══════════════════════════════════════════════════════════════
#  Step 3 — Evaluate predictions from ~5 trading days ago
# ══════════════════════════════════════════════════════════════
def step_evaluate(today: str) -> pd.DataFrame | None:
    log.info("Step 3: Evaluating past predictions …")

    # Find a prediction file from ~5 trading days ago
    pred_file = _find_prediction_file(today)
    if pred_file is None:
        log.info("  No prediction file old enough to evaluate yet — skipping")
        return None

    pred_date = pred_file.stem
    log.info("  Evaluating predictions from %s", pred_date)

    preds = pd.read_csv(pred_file)
    eval_rows = []

    for _, row in preds.iterrows():
        ticker = row["ticker"]
        model = row["model"]
        origin_price = row["origin_price"]
        predicted = np.array([row[f"h{h+1}"] for h in range(HORIZON)])

        # Get actual prices for the HORIZON days after the prediction date
        df = load_cached(ticker)
        close = df["Close"].values.astype(np.float64)
        dates = df.index

        # Find the origin date (last trading day on or before prediction_date)
        pred_dt = pd.Timestamp(pred_date)
        origin_idx = dates.searchsorted(pred_dt, side="right") - 1
        if origin_idx < 0:
            continue

        # Need HORIZON days of actuals after origin
        if origin_idx + HORIZON >= len(close):
            log.info("  %s: not enough actual data after %s — skipping", ticker, pred_date)
            continue

        actuals = close[origin_idx + 1 : origin_idx + 1 + HORIZON]
        context = close[max(0, origin_idx - CONTEXT_LEN + 1) : origin_idx + 1]

        metrics = compute_window_metrics(predicted, actuals, context, origin_price)

        # Direction signal for 1-day-ahead
        signal = 1.0 if predicted[0] > origin_price else -1.0
        actual_ret = (actuals[0] - origin_price) / origin_price

        eval_rows.append({
            "prediction_date": pred_date,
            "evaluation_date": today,
            "ticker": ticker,
            "model": model,
            "mae": round(metrics["mae"], 4),
            "rmse": round(metrics["rmse"], 4),
            "mase": round(metrics["mase"], 4),
            "dir_acc": round(metrics["dir_acc"], 4),
            "signal_1d": signal,
            "return_1d": round(actual_ret, 6),
        })

    if not eval_rows:
        log.info("  No evaluations could be computed")
        return None

    eval_df = pd.DataFrame(eval_rows)

    # Append to cumulative CSV
    if CUMULATIVE_CSV.exists():
        existing = pd.read_csv(CUMULATIVE_CSV)
        # Avoid duplicate evaluations for same prediction_date
        existing = existing[existing["prediction_date"] != pred_date]
        combined = pd.concat([existing, eval_df], ignore_index=True)
    else:
        combined = eval_df

    combined.to_csv(CUMULATIVE_CSV, index=False)
    log.info("  Cumulative results: %d rows → %s", len(combined), CUMULATIVE_CSV.name)

    # Print summary for this evaluation
    summary = eval_df.groupby("model")[["mae", "rmse", "mase", "dir_acc"]].mean()
    log.info("  Evaluation summary for predictions from %s:\n%s",
             pred_date, summary.round(4).to_string())

    return eval_df


def _find_prediction_file(today: str) -> Path | None:
    """Find the prediction file from approximately 5 trading days ago."""
    today_dt = pd.Timestamp(today)

    # List all prediction files, sorted
    pred_files = sorted(PRED_DIR.glob("????-??-??.csv"))
    if not pred_files:
        return None

    # We need at least 5 trading days of actuals.
    # Use SPY's trading calendar to count trading days.
    df = load_cached("SPY")
    if df is None:
        return None
    trading_dates = df.index

    # Find today's position in the trading calendar
    today_pos = trading_dates.searchsorted(today_dt, side="right") - 1
    if today_pos < HORIZON:
        return None

    # The prediction that we can NOW fully evaluate was made on the trading
    # day that is HORIZON trading days before today.
    target_date = trading_dates[today_pos - HORIZON]

    # Find the closest prediction file on or before target_date
    best = None
    for pf in pred_files:
        file_date = pd.Timestamp(pf.stem)
        if file_date <= target_date:
            best = pf

    return best


# ══════════════════════════════════════════════════════════════
#  Step 4 — Generate plots
# ══════════════════════════════════════════════════════════════
def step_plot(today_preds: pd.DataFrame):
    log.info("Step 4: Generating plots …")
    sns.set_theme(style="whitegrid", palette="colorblind")

    _plot_latest_forecasts(today_preds)
    _plot_cumulative_metrics()

    log.info("  Plots saved to %s/", PLOT_DIR)


def _plot_latest_forecasts(preds: pd.DataFrame):
    """Fan chart of today's forecasts for a few representative tickers."""
    sample_tickers = ["NVDA", "AAPL", "SPY", "SOXX"]
    models_to_show = FM_NAMES + ["XGBoost", "Ens-DynWt"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, ticker in zip(axes, sample_tickers):
        sub = preds[preds["ticker"] == ticker]
        origin = sub["origin_price"].iloc[0]

        # Plot a few days of recent history
        df = load_cached(ticker)
        recent = df["Close"].values[-20:]
        x_hist = range(-len(recent), 0)
        ax.plot(x_hist, recent, color="black", linewidth=1.5, label="History")
        ax.scatter([0], [origin], color="black", zorder=5, s=40)

        x_fwd = range(1, HORIZON + 1)
        for model_name in models_to_show:
            row = sub[sub["model"] == model_name]
            if row.empty:
                continue
            vals = [row[f"h{h+1}"].values[0] for h in range(HORIZON)]
            style = "--" if model_name.startswith("Ens") else "-"
            ax.plot(x_fwd, vals, style, linewidth=1.5, label=model_name)

        ax.set_title(ticker, fontsize=13, fontweight="bold")
        ax.set_xlabel("Days from origin")
        ax.set_ylabel("Price ($)")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Forecasts — {preds['prediction_date'].iloc[0]}", fontsize=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "latest_forecasts.png", dpi=150)
    plt.close(fig)


def _plot_cumulative_metrics():
    """Plot running metrics over time from cumulative evaluations."""
    if not CUMULATIVE_CSV.exists():
        log.info("  No cumulative data yet — skipping metric plots")
        return

    cum = pd.read_csv(CUMULATIVE_CSV)
    if cum.empty or len(cum["prediction_date"].unique()) < 2:
        log.info("  Not enough cumulative data for trend plots — skipping")
        return

    cum["prediction_date"] = pd.to_datetime(cum["prediction_date"])

    # ── Model comparison bar chart ───────────────────────────
    avg = cum.groupby("model")[["mae", "rmse", "mase", "dir_acc"]].mean()
    avg = avg.reindex([m for m in ALL_MODEL_NAMES if m in avg.index])

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, col, title in zip(
        axes,
        ["mae", "rmse", "mase", "dir_acc"],
        ["MAE ($)", "RMSE ($)", "MASE", "Directional Accuracy"],
    ):
        colors = []
        for m in avg.index:
            if m in FM_NAMES:
                colors.append("steelblue")
            elif m == "XGBoost":
                colors.append("coral")
            else:
                colors.append("seagreen")
        ax.barh(avg.index, avg[col], color=colors)
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()
    fig.suptitle("Model Comparison (cumulative)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    # ── Running MAE over time ────────────────────────────────
    pivot = cum.pivot_table(
        index="prediction_date", columns="model", values="mae", aggfunc="mean"
    )
    pivot = pivot[[c for c in ALL_MODEL_NAMES if c in pivot.columns]]

    if len(pivot) >= 2:
        fig, ax = plt.subplots(figsize=(12, 5))
        running = pivot.expanding().mean()
        for col in running.columns:
            style = "--" if col.startswith("Ens") else "-"
            ax.plot(running.index, running[col], style, linewidth=1.5, label=col)
        ax.set_ylabel("Cumulative Mean MAE ($)")
        ax.set_title("Running MAE Over Time")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "cumulative_mae.png", dpi=150)
        plt.close(fig)

    # ── Running directional accuracy ─────────────────────────
    pivot_da = cum.pivot_table(
        index="prediction_date", columns="model", values="dir_acc", aggfunc="mean"
    )
    pivot_da = pivot_da[[c for c in ALL_MODEL_NAMES if c in pivot_da.columns]]

    if len(pivot_da) >= 2:
        fig, ax = plt.subplots(figsize=(12, 5))
        running_da = pivot_da.expanding().mean()
        for col in running_da.columns:
            style = "--" if col.startswith("Ens") else "-"
            ax.plot(running_da.index, running_da[col], style, linewidth=1.5, label=col)
        ax.set_ylabel("Cumulative Dir. Accuracy")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="50% baseline")
        ax.set_title("Running Directional Accuracy Over Time")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "directional_accuracy.png", dpi=150)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    today = dt.date.today().isoformat()
    log.info("=" * 60)
    log.info("  Daily pipeline — %s", today)
    log.info("=" * 60)

    # 1. Fetch latest data
    step_fetch()

    # 2. Run predictions
    pred_df = step_predict(today)

    # 3. Evaluate old predictions
    step_evaluate(today)

    # 4. Generate plots
    step_plot(pred_df)

    log.info("=" * 60)
    log.info("  Pipeline complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

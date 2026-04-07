"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(predicted - actual)))


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def mase(
    predicted: np.ndarray,
    actual: np.ndarray,
    history: np.ndarray,
) -> float:
    """Mean Absolute Scaled Error vs one-step naive on in-sample data.

    Scaling factor is the mean absolute one-step difference in the history,
    which is the in-sample MAE of a naive (random-walk) forecast.
    """
    naive_errors = np.abs(np.diff(history))
    scale = np.mean(naive_errors)
    if scale < 1e-10:
        return np.nan
    return float(np.mean(np.abs(predicted - actual)) / scale)


def directional_accuracy(
    predicted: np.ndarray,
    actual: np.ndarray,
    origin_price: float,
) -> float:
    """Fraction of horizon steps where the predicted direction (up/down from
    origin) matches the actual direction."""
    pred_dir = np.sign(predicted - origin_price)
    actual_dir = np.sign(actual - origin_price)
    # Treat zero-move as correct only if both agree
    return float(np.mean(pred_dir == actual_dir))


def sharpe_ratio(
    daily_signals: np.ndarray,
    daily_actual_returns: np.ndarray,
    annualisation: float = 252.0,
) -> float:
    """Annualised Sharpe ratio for a long/short strategy.

    Parameters
    ----------
    daily_signals : array of +1 / -1 (one per evaluation day)
        +1 if the model predicted price going up, -1 if down.
    daily_actual_returns : array of actual daily returns (same length)
    annualisation : trading days per year
    """
    strat_returns = daily_signals * daily_actual_returns
    mu = np.mean(strat_returns)
    sigma = np.std(strat_returns, ddof=1)
    if sigma < 1e-10:
        return 0.0
    return float(mu / sigma * np.sqrt(annualisation))


def compute_window_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    history: np.ndarray,
    origin_price: float,
) -> dict[str, float]:
    """Compute all per-window metrics for a single forecast."""
    return {
        "mae": mae(predicted, actual),
        "rmse": rmse(predicted, actual),
        "mase": mase(predicted, actual, history),
        "dir_acc": directional_accuracy(predicted, actual, origin_price),
    }

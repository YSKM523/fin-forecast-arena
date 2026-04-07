"""Ensemble methods combining foundation model forecasts."""

from __future__ import annotations

import numpy as np

from evaluation.metrics import mae


def equal_weight(forecasts: list[np.ndarray]) -> np.ndarray:
    """Simple average of point forecasts."""
    return np.mean(forecasts, axis=0)


def dynamic_weighted(
    forecasts: list[np.ndarray],
    recent_maes: list[float],
) -> np.ndarray:
    """Weighted average where weights are inverse of each model's recent MAE.

    Parameters
    ----------
    forecasts : list of arrays, one per model, each shape (horizon,)
    recent_maes : list of floats, each model's MAE over the trailing window
    """
    maes = np.array(recent_maes, dtype=np.float64)
    # Guard against zero or near-zero MAE
    maes = np.clip(maes, 1e-8, None)
    inv = 1.0 / maes
    weights = inv / inv.sum()
    return np.average(forecasts, axis=0, weights=weights)


def majority_vote_direction(
    forecasts: list[np.ndarray],
    origin_price: float,
) -> np.ndarray:
    """Majority-vote on direction per horizon step.

    For each step, the predicted direction is the majority vote among models.
    The magnitude comes from the mean of models that agree with the majority.
    Falls back to equal-weight average if the vote is tied (can't happen with 3).
    """
    n_models = len(forecasts)
    horizon = len(forecasts[0])
    result = np.empty(horizon, dtype=np.float64)

    for h in range(horizon):
        votes = [np.sign(f[h] - origin_price) for f in forecasts]
        majority = np.sign(sum(votes))  # +1 or -1 with 3 models

        if majority == 0:
            # Exact tie (unlikely with 3): use equal weight
            result[h] = np.mean([f[h] for f in forecasts])
        else:
            # Average forecasts that agree with majority direction
            agreeing = [f[h] for f, v in zip(forecasts, votes) if v == majority]
            result[h] = np.mean(agreeing) if agreeing else np.mean([f[h] for f in forecasts])

    return result


def compute_trailing_maes(
    predictions: dict[str, list[np.ndarray]],
    actuals: list[np.ndarray],
    window_idx: int,
    lookback: int = 20,
    model_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute each model's MAE over the trailing `lookback` windows.

    Parameters
    ----------
    predictions : {model_name: list of forecast arrays, one per window}
    actuals : list of actual arrays, one per window
    window_idx : current window (exclusive upper bound for lookback)
    lookback : number of past windows to consider
    model_names : models to compute MAE for

    Returns
    -------
    {model_name: trailing_mae}
    """
    if model_names is None:
        model_names = list(predictions.keys())

    start = max(0, window_idx - lookback)
    if start == window_idx:
        # No history yet — return equal MAEs so weights are equal
        return {m: 1.0 for m in model_names}

    result = {}
    for m in model_names:
        errors = []
        for i in range(start, window_idx):
            errors.append(mae(predictions[m][i], actuals[i]))
        result[m] = float(np.mean(errors))

    return result

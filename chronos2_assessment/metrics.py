"""Forecast evaluation metrics: MASE, WQL, 80% interval coverage, MAPE."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season_length: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Scaled by in-sample seasonal-naive MAE on `y_train`. Values <1 beat naive.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= season_length:
        raise ValueError("y_train shorter than season_length")
    scale = np.mean(np.abs(y_train[season_length:] - y_train[:-season_length]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def weighted_quantile_loss(
    y_true: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
) -> float:
    """Weighted Quantile Loss across the supplied quantile levels.

    `quantile_forecasts` maps level (e.g. 0.1, 0.5, 0.9) to predicted values.
    Normalised by sum(|y_true|).
    """
    y_true = np.asarray(y_true, dtype=float)
    total = 0.0
    for q, yhat in quantile_forecasts.items():
        yhat = np.asarray(yhat, dtype=float)
        diff = y_true - yhat
        total += 2.0 * np.sum(np.maximum(q * diff, (q - 1) * diff))
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(total / (len(quantile_forecasts) * denom))


def interval_coverage(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> float:
    """Fraction of `y_true` falling in `[q_low, q_high]`. For 80% PI should ~= 0.8."""
    y_true = np.asarray(y_true, dtype=float)
    q_low = np.asarray(q_low, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    return float(np.mean((y_true >= q_low) & (y_true <= q_high)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Pathological near zero; report as secondary."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def score_forecast(
    forecast: pd.DataFrame,
    actuals: pd.DataFrame,
    y_train: np.ndarray,
    season_length: int,
) -> dict[str, float]:
    """Compute the full metric panel for one method's forecast vs actuals.

    `forecast` columns: timestamp, q10, q50, q90. `actuals` columns: timestamp, target.
    """
    merged = forecast.merge(actuals, on="timestamp", how="inner")
    y_true = merged["target"].to_numpy()
    q10 = merged["q10"].to_numpy()
    q50 = merged["q50"].to_numpy()
    q90 = merged["q90"].to_numpy()
    return {
        "mase": mase(y_true, q50, y_train, season_length=season_length),
        "wql": weighted_quantile_loss(y_true, {0.1: q10, 0.5: q50, 0.9: q90}),
        "coverage_80": interval_coverage(y_true, q10, q90),
        "mape": mape(y_true, q50),
    }

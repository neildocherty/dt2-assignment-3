"""Known-answer tests for forecast metrics."""

from __future__ import annotations

import numpy as np

from chronos2_assessment.metrics import (
    interval_coverage,
    mape,
    mase,
    weighted_quantile_loss,
)


def test_mase_perfect_forecast_is_zero() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    y_train = np.array([5.0, 10.0, 15.0, 20.0])
    assert mase(y_true, y_pred, y_train) == 0.0


def test_mase_matches_naive() -> None:
    # naive forecast (constant 10) on y_true=[20,30] with seasonal-1 scale of 5 -> MASE = 15/5 = 3
    y_true = np.array([20.0, 30.0])
    y_pred = np.array([10.0, 10.0])
    y_train = np.array([5.0, 10.0, 15.0, 20.0])
    assert mase(y_true, y_pred, y_train) == 3.0


def test_interval_coverage_all_inside() -> None:
    y_true = np.array([5.0, 5.0, 5.0])
    q_low = np.array([0.0, 0.0, 0.0])
    q_high = np.array([10.0, 10.0, 10.0])
    assert interval_coverage(y_true, q_low, q_high) == 1.0


def test_interval_coverage_partial() -> None:
    y_true = np.array([1.0, 5.0, 11.0])
    q_low = np.array([0.0, 0.0, 0.0])
    q_high = np.array([10.0, 10.0, 10.0])
    assert interval_coverage(y_true, q_low, q_high) == 2 / 3


def test_wql_perfect_median_is_zero() -> None:
    y_true = np.array([10.0, 20.0])
    quantile_forecasts = {0.5: np.array([10.0, 20.0])}
    assert weighted_quantile_loss(y_true, quantile_forecasts) == 0.0


def test_mape_perfect_forecast_is_zero() -> None:
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([10.0, 20.0])
    assert mape(y_true, y_pred) == 0.0

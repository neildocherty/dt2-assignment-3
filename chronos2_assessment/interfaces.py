"""Unified forecasting interface shared by all baselines and Chronos-2 runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class ForecastResult:
    """Quantile forecast with 10th / 50th / 90th percentile bands.

    `frame` columns: ``[timestamp, q10, q50, q90]``. ``q50`` is the point forecast;
    ``q10``/``q90`` bound the 80% prediction interval used for coverage scoring.
    """

    frame: pd.DataFrame
    method: str

    def __post_init__(self) -> None:
        required = {"timestamp", "q10", "q50", "q90"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"ForecastResult.frame missing columns: {missing}")


class Forecaster(Protocol):
    """Protocol every forecasting method must satisfy.

    Implementations are stateful: `fit` ingests training history, `predict` returns
    quantile forecasts over a horizon, optionally informed by future covariates.
    """

    name: str

    def fit(self, train: pd.DataFrame) -> None: ...

    def predict(
        self,
        horizon: int,
        future_covariates: pd.DataFrame | None = None,
    ) -> ForecastResult: ...

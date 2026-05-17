"""Chronos-2 zero-shot forecasting wrappers, with and without covariates."""

from __future__ import annotations

import pandas as pd

from .baselines import WEATHER_COLS
from .interfaces import ForecastResult

_PIPELINE_CACHE: dict[tuple[str, str], object] = {}


def _get_pipeline(model_id: str, device: str):
    from chronos import Chronos2Pipeline

    key = (model_id, device)
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = Chronos2Pipeline.from_pretrained(model_id, device_map=device)
    return _PIPELINE_CACHE[key]


def _to_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pred_df["timestamp"]).to_numpy(),
            "q10": pred_df["0.1"].to_numpy(),
            "q50": pred_df["0.5"].to_numpy(),
            "q90": pred_df["0.9"].to_numpy(),
        }
    )


class ChronosUnivariate:
    """Chronos-2 zero-shot on the target series only — no covariates."""

    name = "chronos2_univariate"

    def __init__(self, model_id: str = "amazon/chronos-2", device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device

    def fit(self, train: pd.DataFrame) -> None:
        self.context = train[["id", "timestamp", "target"]].copy()

    def predict(
        self,
        horizon: int,
        future_covariates: pd.DataFrame | None = None,
    ) -> ForecastResult:
        pipeline = _get_pipeline(self.model_id, self.device)
        pred = pipeline.predict_df(
            self.context,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        return ForecastResult(frame=_to_frame(pred), method=self.name)


class ChronosWithCovariates:
    """Chronos-2 zero-shot with weather covariates — the v2 headline capability."""

    name = "chronos2_covariates"

    def __init__(self, model_id: str = "amazon/chronos-2", device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device

    def fit(self, train: pd.DataFrame) -> None:
        keep = ["id", "timestamp", "target", *WEATHER_COLS]
        self.context = train[keep].copy()

    def predict(
        self,
        horizon: int,
        future_covariates: pd.DataFrame | None = None,
    ) -> ForecastResult:
        assert future_covariates is not None, "weather covariates required"
        pipeline = _get_pipeline(self.model_id, self.device)
        future_df = future_covariates[["id", "timestamp", *WEATHER_COLS]].copy()
        pred = pipeline.predict_df(
            self.context,
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        return ForecastResult(frame=_to_frame(pred), method=self.name)

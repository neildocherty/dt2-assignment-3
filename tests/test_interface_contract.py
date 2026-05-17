"""Each forecaster must conform to the `Forecaster` protocol and emit the right schema."""

from __future__ import annotations

import pytest

from chronos2_assessment.baselines import (
    AutoARIMA,
    LightGBMForecaster,
    ProphetForecaster,
    SeasonalNaive,
)
from chronos2_assessment.chronos_runner import ChronosUnivariate, ChronosWithCovariates
from chronos2_assessment.interfaces import Forecaster

ALL_FORECASTERS = [
    SeasonalNaive,
    AutoARIMA,
    lambda: ProphetForecaster(use_weather=False),
    lambda: ProphetForecaster(use_weather=True),
    LightGBMForecaster,
    ChronosUnivariate,
    ChronosWithCovariates,
]


@pytest.mark.parametrize("ctor", ALL_FORECASTERS)
def test_forecaster_has_name(ctor) -> None:
    instance = ctor()
    assert isinstance(instance.name, str) and instance.name


@pytest.mark.parametrize("ctor", ALL_FORECASTERS)
def test_forecaster_satisfies_protocol(ctor) -> None:
    instance: Forecaster = ctor()
    assert hasattr(instance, "fit")
    assert hasattr(instance, "predict")

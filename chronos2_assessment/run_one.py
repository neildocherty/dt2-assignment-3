"""Run one forecaster end-to-end in an isolated subprocess.

Each call is a fresh Python process — avoids the LightGBM ↔ PyTorch OpenMP
deadlock we hit when loading both into the same kernel on macOS.

Usage: ``python -m chronos2_assessment.run_one <method> <out_dir>``

Writes ``<out_dir>/<method>.parquet`` (timestamp, q10, q50, q90) and
``<out_dir>/<method>.json`` (metric panel + runtime).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from .baselines import (  # noqa: E402
    CALENDAR_COLS,
    WEATHER_COLS,
    AutoARIMA,
    LightGBMForecaster,
    ProphetForecaster,
    SeasonalNaive,
)
from .chronos_runner import ChronosUnivariate, ChronosWithCovariates  # noqa: E402
from .data_loader import load_processed  # noqa: E402
from .metrics import score_forecast  # noqa: E402

HORIZON = 24 * 7
SEASON = 24

REGISTRY = {
    "seasonal_naive": (lambda: SeasonalNaive(), False),
    "auto_arima": (lambda: AutoARIMA(season_length=SEASON, context_length=24 * 30), False),
    "prophet": (lambda: ProphetForecaster(use_weather=False), False),
    "prophet_weather": (lambda: ProphetForecaster(use_weather=True), True),
    "lightgbm": (lambda: LightGBMForecaster(), True),
    "chronos2_univariate": (lambda: ChronosUnivariate(), False),
    "chronos2_covariates": (lambda: ChronosWithCovariates(), True),
}


def main(method: str, out_dir: Path, processed_dir: Path) -> None:
    ctor, needs_cov = REGISTRY[method]
    df = load_processed(processed_dir)
    train = df.iloc[:-HORIZON].copy()
    test = df.iloc[-HORIZON:].copy()
    future_cov = test[["timestamp", "id"] + WEATHER_COLS + CALENDAR_COLS].copy()
    actuals = test[["timestamp", "target"]].copy()
    y_train = train["target"].to_numpy()

    t0 = time.time()
    model = ctor()
    model.fit(train)
    result = model.predict(HORIZON, future_cov if needs_cov else None)
    elapsed = time.time() - t0

    scores = score_forecast(result.frame, actuals, y_train, season_length=SEASON)
    scores["method"] = method
    scores["runtime_s"] = round(elapsed, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    result.frame.to_parquet(out_dir / f"{method}.parquet", index=False)
    (out_dir / f"{method}.json").write_text(json.dumps(scores, default=float, indent=2))
    print(f"{method} {elapsed:.1f}s", scores, flush=True)


if __name__ == "__main__":
    main(
        method=sys.argv[1],
        out_dir=Path(sys.argv[2]),
        processed_dir=Path(sys.argv[3]),
    )

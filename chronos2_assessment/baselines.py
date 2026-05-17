"""Classical and ML baselines, all conforming to the `Forecaster` protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .interfaces import ForecastResult

WEATHER_COLS = ["temp", "atemp", "hum", "windspeed", "weather"]
CALENDAR_COLS = ["hour", "weekday", "workingday", "holiday"]


def _future_timestamps(last_ts: pd.Timestamp, horizon: int, freq: str = "h") -> pd.DatetimeIndex:
    last_ts = pd.Timestamp(last_ts).as_unit("ns")
    return pd.date_range(start=last_ts + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)


def _align_covariates(future_covariates: pd.DataFrame, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex future_covariates onto `ts`, coercing timestamps to ns."""
    cov = future_covariates.copy()
    cov["timestamp"] = pd.to_datetime(cov["timestamp"]).astype("datetime64[ns]")
    return cov.set_index("timestamp").reindex(ts)


class SeasonalNaive:
    """Floor baseline: repeat target from `season_length` steps ago.

    Quantile bands estimated from in-sample residuals against the same naive rule.
    """

    name = "seasonal_naive"

    def __init__(self, season_length: int = 24 * 7) -> None:
        self.season_length = season_length

    def fit(self, train: pd.DataFrame) -> None:
        self.train = train.sort_values("timestamp").reset_index(drop=True)
        y = self.train["target"].to_numpy(dtype=float)
        resid = y[self.season_length :] - y[: -self.season_length]
        self.resid_std = float(np.std(resid))

    def predict(
        self,
        horizon: int,
        future_covariates: pd.DataFrame | None = None,
    ) -> ForecastResult:
        y = self.train["target"].to_numpy(dtype=float)
        last_ts = self.train["timestamp"].iloc[-1]
        # repeat last seasonal window across horizon
        season = y[-self.season_length :]
        reps = int(np.ceil(horizon / self.season_length))
        q50 = np.tile(season, reps)[:horizon]
        # normal-approx PI from in-sample residual std
        z = 1.2816  # 80% PI
        q10 = q50 - z * self.resid_std
        q90 = q50 + z * self.resid_std
        ts = _future_timestamps(last_ts, horizon)
        frame = pd.DataFrame({"timestamp": ts, "q10": q10, "q50": q50, "q90": q90})
        return ForecastResult(frame=frame, method=self.name)


class AutoARIMA:
    """Classical ARIMA via statsforecast.AutoARIMA on hourly series."""

    name = "auto_arima"

    def __init__(self, season_length: int = 24, context_length: int = 24 * 60) -> None:
        self.season_length = season_length
        self.context_length = context_length

    def fit(self, train: pd.DataFrame) -> None:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA as SFAutoARIMA

        truncated = train.tail(self.context_length)
        df = truncated[["timestamp", "target"]].rename(
            columns={"timestamp": "ds", "target": "y"}
        )
        df["unique_id"] = "series"
        self.sf = StatsForecast(
            models=[SFAutoARIMA(season_length=self.season_length)],
            freq="h",
            n_jobs=1,
        )
        self.sf.fit(df)
        self.last_ts = train["timestamp"].iloc[-1]

    def predict(
        self,
        horizon: int,
        future_covariates: pd.DataFrame | None = None,
    ) -> ForecastResult:
        fc = self.sf.predict(h=horizon, level=[80])
        ts = _future_timestamps(self.last_ts, horizon)
        q50 = fc["AutoARIMA"].to_numpy()
        q10 = fc["AutoARIMA-lo-80"].to_numpy()
        q90 = fc["AutoARIMA-hi-80"].to_numpy()
        frame = pd.DataFrame({"timestamp": ts, "q10": q10, "q50": q50, "q90": q90})
        return ForecastResult(frame=frame, method=self.name)


class ProphetForecaster:
    """Facebook Prophet. `use_weather=True` adds weather columns as regressors."""

    def __init__(self, use_weather: bool = False) -> None:
        self.use_weather = use_weather
        self.name = "prophet_weather" if use_weather else "prophet"

    def fit(self, train: pd.DataFrame) -> None:
        from prophet import Prophet

        df = train.rename(columns={"timestamp": "ds", "target": "y"})
        self.model = Prophet(
            interval_width=0.8,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
        )
        if self.use_weather:
            for c in WEATHER_COLS:
                self.model.add_regressor(c)
        self.model.fit(df)
        self.last_ts = train["timestamp"].iloc[-1]

    def predict(
        self,
        horizon: int,
        future_covariates: pd.DataFrame | None = None,
    ) -> ForecastResult:
        ts = _future_timestamps(self.last_ts, horizon)
        future = pd.DataFrame({"ds": ts})
        if self.use_weather:
            assert future_covariates is not None, "weather covariates required"
            cov = _align_covariates(future_covariates, ts)
            for c in WEATHER_COLS:
                future[c] = cov[c].to_numpy()
        fc = self.model.predict(future)
        frame = pd.DataFrame(
            {
                "timestamp": fc["ds"],
                "q10": fc["yhat_lower"].to_numpy(),
                "q50": fc["yhat"].to_numpy(),
                "q90": fc["yhat_upper"].to_numpy(),
            }
        )
        return ForecastResult(frame=frame, method=self.name)


class LightGBMForecaster:
    """LightGBM with engineered lag, calendar, and weather features.

    Quantile bands produced by training three models at alpha=0.1/0.5/0.9 with
    the ``quantile`` objective. Recursive multi-step prediction using lag-1/24/168.
    """

    name = "lightgbm"

    LAGS = (1, 24, 168)

    def __init__(self) -> None:
        self.models: dict[float, object] = {}

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        for c in CALENDAR_COLS + WEATHER_COLS:
            feats[c] = df[c].to_numpy()
        for lag in self.LAGS:
            feats[f"lag_{lag}"] = df["target"].shift(lag).to_numpy()
        return feats

    def fit(self, train: pd.DataFrame) -> None:
        import lightgbm as lgb

        df = train.sort_values("timestamp").reset_index(drop=True)
        X = self._features(df).dropna()
        y = df.loc[X.index, "target"].astype(float)
        params = dict(objective="quantile", n_estimators=400, learning_rate=0.05, verbosity=-1)
        for q in (0.1, 0.5, 0.9):
            m = lgb.LGBMRegressor(alpha=q, **params)
            m.fit(X, y)
            self.models[q] = m
        self.history = df.copy()

    def predict(
        self,
        horizon: int,
        future_covariates: pd.DataFrame | None = None,
    ) -> ForecastResult:
        assert future_covariates is not None, "covariates required (calendar+weather)"
        hist = self.history.copy()
        last_ts = hist["timestamp"].iloc[-1]
        ts = _future_timestamps(last_ts, horizon)
        future = _align_covariates(future_covariates, ts).reset_index().rename(
            columns={"index": "timestamp"}
        )
        future["timestamp"] = ts
        out = {0.1: [], 0.5: [], 0.9: []}
        for i in range(horizon):
            row_df = pd.concat([hist, future.iloc[[i]].assign(target=np.nan)], ignore_index=True)
            feats = self._features(row_df).iloc[[-1]]
            preds = {q: float(self.models[q].predict(feats)[0]) for q in (0.1, 0.5, 0.9)}
            for q in preds:
                out[q].append(preds[q])
            new_row = future.iloc[[i]].copy()
            new_row["target"] = preds[0.5]
            hist = pd.concat([hist, new_row], ignore_index=True)
        frame = pd.DataFrame(
            {"timestamp": ts, "q10": out[0.1], "q50": out[0.5], "q90": out[0.9]}
        )
        # ensure monotone quantiles
        frame["q10"] = np.minimum(frame["q10"], frame["q50"])
        frame["q90"] = np.maximum(frame["q90"], frame["q50"])
        return ForecastResult(frame=frame, method=self.name)

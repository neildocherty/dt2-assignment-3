"""UCI Bike Sharing download + reshape to long-format parquet.

Source: <https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset>
Hourly resolution, 2011-01-01 → 2012-12-31, ~17k rows, single series.

Processed schema (one row per hour):
    id           str    constant "bike_share"
    timestamp    datetime
    target       int    hourly rental count (cnt)
    temp         float  normalised temperature [0,1] (raw / 41 °C)
    atemp        float  normalised "feels-like" temperature [0,1] (raw / 50 °C)
    hum          float  normalised humidity [0,1]
    windspeed    float  normalised wind speed [0,1]
    weather      int    weather situation 1=clear … 4=heavy rain/snow
    hour         int    0–23
    weekday      int    0=Sun … 6=Sat
    workingday   int    1 if not weekend/holiday
    holiday      int    1 if holiday
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

UCI_DATASET_ID = 275
SERIES_ID = "bike_share"
PROCESSED_FILENAME = "bike_share_hourly.parquet"


def fetch_raw() -> pd.DataFrame:
    """Pull the raw UCI dataset and join features + target into one frame."""
    ds = fetch_ucirepo(id=UCI_DATASET_ID)
    return pd.concat([ds.data.features, ds.data.targets], axis=1)


def reshape(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw UCI rows to long format, reindexed onto a complete hourly grid.

    UCI Bike Sharing has ~165 missing hours; we fill with interpolation (weather)
    and time-aware imputation (target / calendar) so downstream models see a regular
    hourly series.
    """
    timestamp = pd.to_datetime(raw["dteday"]) + pd.to_timedelta(raw["hr"], unit="h")
    timestamp = pd.DatetimeIndex(timestamp).astype("datetime64[ns]")
    out = pd.DataFrame(
        {
            "id": SERIES_ID,
            "timestamp": timestamp,
            "target": raw["cnt"].astype("float64"),
            "temp": raw["temp"].astype("float32"),
            "atemp": raw["atemp"].astype("float32"),
            "hum": raw["hum"].astype("float32"),
            "windspeed": raw["windspeed"].astype("float32"),
            "weather": raw["weathersit"].astype("int8"),
            "hour": raw["hr"].astype("int8"),
            "weekday": raw["weekday"].astype("int8"),
            "workingday": raw["workingday"].astype("int8"),
            "holiday": raw["holiday"].astype("int8"),
        }
    ).sort_values("timestamp").reset_index(drop=True)

    full = pd.date_range(out["timestamp"].min(), out["timestamp"].max(), freq="h")
    out = out.set_index("timestamp").reindex(full).rename_axis("timestamp").reset_index()
    out["id"] = SERIES_ID
    # interpolate weather + target; rederive calendar from timestamp
    for c in ("temp", "atemp", "hum", "windspeed"):
        out[c] = out[c].interpolate("linear").bfill().ffill().astype("float32")
    out["weather"] = out["weather"].ffill().bfill().astype("int8")
    out["target"] = out["target"].interpolate("linear").bfill().ffill().round().astype("int64")
    out["hour"] = out["timestamp"].dt.hour.astype("int8")
    out["weekday"] = out["timestamp"].dt.dayofweek.astype("int8")  # 0=Mon
    out["holiday"] = out["holiday"].ffill().bfill().astype("int8")
    out["workingday"] = out["workingday"].ffill().bfill().astype("int8")
    return out


def build_processed(out_dir: Path) -> Path:
    """Download → reshape → persist parquet. Returns the parquet path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = reshape(fetch_raw())
    out_path = out_dir / PROCESSED_FILENAME
    df.to_parquet(out_path, index=False)
    return out_path


def load_processed(processed_dir: Path) -> pd.DataFrame:
    """Load the processed parquet. Builds it on first call if missing."""
    path = processed_dir / PROCESSED_FILENAME
    if not path.exists():
        build_processed(processed_dir)
    return pd.read_parquet(path)

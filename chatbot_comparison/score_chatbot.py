"""Score a chatbot CSV reply against actuals; persist alongside model outputs.

Usage: python chatbot_comparison/score_chatbot.py <name> <csv_path>

The CSV must have columns: timestamp, q10, q50, q90 (168 rows).
Writes outputs/forecasts/<name>.{parquet,json} so the next notebook run picks it up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from chronos2_assessment.data_loader import load_processed
from chronos2_assessment.metrics import score_forecast

HORIZON = 24 * 7
SEASON = 24


def main(name: str, csv_path: Path) -> None:
    df = load_processed(Path("data/processed"))
    train, test = df.iloc[:-HORIZON], df.iloc[-HORIZON:]
    actuals = test[["timestamp", "target"]].copy()
    y_train = train["target"].to_numpy()

    fc = pd.read_csv(csv_path)
    fc["timestamp"] = pd.to_datetime(fc["timestamp"]).astype("datetime64[ns]")
    missing = set(["q10", "q50", "q90"]) - set(fc.columns)
    assert not missing, f"CSV missing columns: {missing}"
    assert len(fc) == HORIZON, f"expected {HORIZON} rows, got {len(fc)}"

    scores = score_forecast(fc, actuals, y_train, season_length=SEASON)
    scores["method"] = name
    scores["runtime_s"] = None

    out_dir = Path("outputs/forecasts")
    out_dir.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / f"{name}.json").write_text(json.dumps(scores, default=float, indent=2))
    print(json.dumps(scores, default=float, indent=2))


if __name__ == "__main__":
    main(name=sys.argv[1], csv_path=Path(sys.argv[2]))

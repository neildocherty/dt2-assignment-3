"""Generate prompt.md with last-2-weeks context + next-168h covariates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chronos2_assessment.data_loader import load_processed

HORIZON = 24 * 7
CONTEXT = 24 * 14


def main() -> None:
    df = load_processed(Path("data/processed"))
    train = df.iloc[:-HORIZON]
    test = df.iloc[-HORIZON:]
    ctx = train.tail(CONTEXT)[
        ["timestamp", "target", "temp", "atemp", "hum", "windspeed", "weather", "hour", "weekday", "workingday", "holiday"]
    ]
    future_cov = test[
        ["timestamp", "temp", "atemp", "hum", "windspeed", "weather", "hour", "weekday", "workingday", "holiday"]
    ]

    ctx_csv = ctx.to_csv(index=False)
    fut_csv = future_cov.to_csv(index=False)
    first_ts = future_cov["timestamp"].iloc[0]
    last_ts = future_cov["timestamp"].iloc[-1]

    body = f"""# Hourly bike-rental forecast — please complete

You are forecasting hourly bike rentals in Washington DC (UCI Bike Sharing Dataset).
Below is the **last 14 days of history** (336 rows) and the **next 7 days of weather + calendar features** (168 rows).

Produce a **168-hour forecast** for `target` (rentals/hour) from {first_ts} through {last_ts}, with **80% prediction intervals**.

Return your answer as CSV with **exactly these columns and 168 rows**:

```
timestamp,q10,q50,q90
```

- `q10`, `q90` are the lower/upper bounds of the 80% prediction interval.
- `q50` is the median (point) forecast.
- Use the timestamps from the future-covariates block below, in order.
- Do not return code; return the CSV directly.

## Notes on the data

- `target`: hourly rental count (integer in history; your q50 may be float).
- `temp`, `atemp`, `hum`, `windspeed`: normalised to [0,1] (raw °C / 41, "feels-like" / 50, humidity / 100, wind / 67).
- `weather`: 1=clear, 2=mist, 3=light rain/snow, 4=heavy.
- `hour`: 0–23. `weekday`: 0=Monday … 6=Sunday. `workingday`/`holiday`: 0/1.
- Strong daily + weekly seasonality; weather and working-day effects are material.

## History (last 14 days, {len(ctx)} rows)

```csv
{ctx_csv}```

## Future covariates ({HORIZON} rows — your forecast horizon)

```csv
{fut_csv}```
"""

    out = Path("chatbot_comparison/prompt.md")
    out.write_text(body)
    print(f"wrote {out} ({len(body):,} chars, ~{len(body)//4:,} tokens)")


if __name__ == "__main__":
    main()

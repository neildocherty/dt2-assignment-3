# Chronos-2 Niche GenAI Assessment

DT2 assignment 3. Tests Amazon's [Chronos-2](https://huggingface.co/amazon/chronos-2) time-series foundation model against classical/ML baselines on UCI Bike Sharing demand, with covariate-informed forecasting.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (installs Python 3.11 automatically).

## Setup

```sh
uv sync
```

Materialises `.venv/` with all runtime + dev deps.

## Run

```sh
# 1. Chronos-2 install sanity check
uv run jupyter lab notebooks/00_sanity_check.ipynb

# 2. EDA
uv run jupyter lab notebooks/01_data_exploration.ipynb

# 3. Full experiment (headless)
uv run jupyter execute notebooks/02_main_experiment.ipynb
```

## Render the HTML report

The submission notebook is `notebooks/03_report.ipynb`. To regenerate the standalone HTML at `outputs/report.html` (max-width-constrained for legibility, with embedded images):

```sh
scripts/render_html.sh
```

The script executes the notebook in place and renders the HTML in one step. To render a different notebook:

```sh
scripts/render_html.sh notebooks/02_main_experiment.ipynb outputs/report.html
```

## Tests

```sh
uv run pytest
```

## Lint & format

```sh
uv run ruff check --fix && uv run ruff format
```

## Layout

- `chronos2_assessment/` — library code (forecasters, metrics, plots, data loading).
- `notebooks/` — `00` sanity check, `01` EDA, `02` working experiment, `03` submission report.
- `tests/` — pytest smoke tests on metric correctness and forecaster interface contract.
- `data/processed/` — committed parquet; `data/raw/` gitignored (downloaded on demand).
- `outputs/figures/` and `outputs/tables/` — committed artifacts.
- `chatbot_comparison/` — manual ChatGPT/Claude transcripts and screenshots.

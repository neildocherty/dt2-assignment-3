"""Publication-quality plots for the submission notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forecast_vs_actual(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    actuals: pd.DataFrame,
    title: str,
    out_path: Path | None = None,
    history_tail: int = 24 * 7,
) -> plt.Figure:
    """Time-series plot: history + forecast median + 80% band + actuals overlay."""
    fig, ax = plt.subplots(figsize=(12, 4))
    h = history.tail(history_tail)
    ax.plot(h["timestamp"], h["target"], color="#1f77b4", label="history")
    ax.plot(actuals["timestamp"], actuals["target"], color="#2ca02c", label="actual")
    ax.plot(forecast["timestamp"], forecast["q50"], color="#9467bd", label="forecast (q50)")
    ax.fill_between(
        forecast["timestamp"], forecast["q10"], forecast["q90"],
        color="#9467bd", alpha=0.25, label="80% band",
    )
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_metric_bars(metrics: pd.DataFrame, out_path: Path | None = None) -> plt.Figure:
    """Bar plot comparing methods across MASE / WQL / coverage."""
    cols = ["mase", "wql", "coverage_80"]
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 4))
    for ax, col in zip(axes, cols, strict=True):
        m = metrics.sort_values(col, ascending=(col != "coverage_80"))
        colors = ["#9467bd" if "chronos" in n else "#7f7f7f" for n in m["method"]]
        ax.barh(m["method"], m[col], color=colors)
        ax.set_title(col)
        if col == "coverage_80":
            ax.axvline(0.8, color="red", linestyle="--", linewidth=1, label="nominal 0.8")
            ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_all_forecasts(
    history: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    actuals: pd.DataFrame,
    out_path: Path | None = None,
    history_tail: int = 24 * 3,
) -> plt.Figure:
    """Grid of forecast-vs-actual plots, one panel per method."""
    n = len(forecasts)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True)
    axes = np.atleast_2d(axes).ravel()
    h = history.tail(history_tail)
    for ax, (name, fc) in zip(axes, forecasts.items(), strict=False):
        ax.plot(h["timestamp"], h["target"], color="#1f77b4", linewidth=1, label="history")
        ax.plot(actuals["timestamp"], actuals["target"], color="#2ca02c", linewidth=1.2, label="actual")
        ax.plot(fc["timestamp"], fc["q50"], color="#9467bd", linewidth=1.2, label="q50")
        ax.fill_between(fc["timestamp"], fc["q10"], fc["q90"], color="#9467bd", alpha=0.2)
        ax.set_title(name, fontsize=10)
    for ax in axes[len(forecasts):]:
        ax.set_visible(False)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=120)
    return fig

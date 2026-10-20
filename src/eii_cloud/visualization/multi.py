"""Multi-signal, context-comparison and detector-comparison charts."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..data import EIIDataset
from ..findings import AnalysisResult
from ..truth import runs_of
from .style import ChartStyle


class MultiSignalChart:
    """Stacked panels of several parameters with findings shaded per panel."""

    def __init__(self, dataset: EIIDataset, result: AnalysisResult | None = None, parameters: list[str] | None = None,
                 style: ChartStyle | None = None):
        self.dataset, self.result = dataset, result
        self.parameters = parameters or dataset.parameters
        self.style = style or ChartStyle(figsize=(14, 2.2 * len(self.parameters) + 1.5))

    def render(self) -> Figure:
        fig, axes = plt.subplots(len(self.parameters), 1, sharex=True, figsize=self.style.figsize, dpi=self.style.dpi,
                                 squeeze=False)
        x = self.dataset.time.plot_values()
        for ax, p in zip(axes[:, 0], self.parameters):
            ax.plot(x, self.dataset.value(p), color=self.style.observed_color, lw=1.2, marker=".", ms=3)
            if self.result is not None:
                env = next((e for e in self.result.envelopes.values() if e["parameter"] == p), None)
                if env is not None:
                    lo, hi = np.asarray(env["lower"], float), np.asarray(env["upper"], float)
                    ax.fill_between(x, lo, hi, color=self.style.cloud_color, alpha=self.style.cloud_alpha, lw=0)
                for f in self.result.by_parameter(p):
                    ax.axvspan(x[f.start_index], x[f.end_index], color=self.style.flag_colors.get(f.category.value, "k"),
                               alpha=0.2, lw=0)
            ax.set_ylabel(p + (f" [{self.dataset.unit(p)}]" if self.dataset.unit(p) else ""))
            ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("Time")
        fig.suptitle("Multi-signal view with findings")
        fig.tight_layout()
        return fig


class ContextComparisonChart:
    """Distribution of a parameter per context regime, with the envelope's per-regime bounds."""

    def __init__(self, dataset: EIIDataset, parameter: str, regimes: np.ndarray, result: AnalysisResult | None = None,
                 envelope: str | None = None, style: ChartStyle | None = None):
        self.dataset, self.parameter, self.regimes = dataset, parameter, np.asarray(regimes).astype(str)
        self.result, self.envelope = result, envelope
        self.style = style or ChartStyle(figsize=(10, 6))

    def render(self, ax: Any = None) -> Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        y = self.dataset.value(self.parameter)
        groups = sorted(set(self.regimes.tolist()))
        data = [y[(self.regimes == g) & ~np.isnan(y)] for g in groups]
        ax.boxplot(data, tick_labels=groups, widths=0.5)
        if self.result is not None:
            name = self.envelope or next((k for k, e in self.result.envelopes.items() if e["parameter"] == self.parameter), None)
            if name is not None:
                e = self.result.envelopes[name]
                lo, hi = np.asarray(e["lower"], float), np.asarray(e["upper"], float)
                for k, g in enumerate(groups, start=1):
                    m = self.regimes == g
                    if np.any(~np.isnan(lo[m])):
                        ax.hlines([np.nanmedian(lo[m]), np.nanmedian(hi[m])], k - 0.35, k + 0.35,
                                  colors=self.style.cloud_color, lw=2)
        ax.set_ylabel(self.parameter)
        ax.set_title(f"{self.parameter} by context regime (bars: median envelope bounds)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        return fig


class DetectorComparisonChart:
    """Raster of truth series (TRUE/FALSE/UNKNOWN) for every detector/rule, optionally with ground truth."""

    def __init__(self, dataset: EIIDataset, result: AnalysisResult, ground_truth: Mapping[str, np.ndarray] | None = None,
                 style: ChartStyle | None = None):
        self.dataset, self.result, self.ground_truth = dataset, result, ground_truth or {}
        n = len(result.truth_series) + len(self.ground_truth)
        self.style = style or ChartStyle(figsize=(14, 0.45 * n + 2.5))

    def render(self, ax: Any = None) -> Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        rows = [(f"truth: {k}", np.asarray(v, dtype=np.int8)) for k, v in self.ground_truth.items()]
        rows += [(k, np.asarray(v, dtype=np.int8)) for k, v in self.result.truth_series.items()]
        x = self.dataset.time.plot_values()
        colors = {1: "#d62728", 0: "#e0e0e0", -1: "#9e9e9e"}
        for r, (_name, v) in enumerate(rows):
            for val, col in colors.items():
                for s, e in runs_of(v == val):
                    w = (x[e] - x[s]) if e > s else (x[1] - x[0] if len(x) > 1 else 1)
                    ax.barh(r, w, left=x[s], height=0.7, color=col, lw=0)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel("Time")
        ax.set_title("Detector comparison (red = TRUE, grey = UNKNOWN, light = FALSE)")
        fig.tight_layout()
        return fig

"""Score, control, CUSUM and change-point charts."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..data import EIIDataset
from ..findings import AnalysisResult
from ..truth import runs_of
from .style import ChartStyle


def _series(result: AnalysisResult, detector_id: str) -> tuple[np.ndarray, np.ndarray]:
    if detector_id not in result.scores:
        raise KeyError(f"no score series for {detector_id!r}; available: {sorted(result.scores)}")
    score = np.array([np.nan if v is None else v for v in result.scores[detector_id]], dtype=float)
    truth = np.asarray(result.truth_series[detector_id], dtype=np.int8)
    return score, truth


class AnomalyScoreChart:
    """Observed signal (top) and a detector's normalised score with its alarm level (bottom)."""

    def __init__(self, dataset: EIIDataset, result: AnalysisResult, detector_id: str, parameter: str | None = None,
                 threshold: float = 1.0, style: ChartStyle | None = None, title: str | None = None):
        self.dataset, self.result, self.detector_id = dataset, result, detector_id
        self.parameter = parameter or next((d["parameter"] for d in result.config.get("detectors", [])
                                            if d.get("detector_id") == detector_id), dataset.parameters[0])
        self.threshold = threshold
        self.style = style or ChartStyle(figsize=(14, 7))
        self.title = title or f"{detector_id}: score and alarms"

    def render(self, axes: Any = None) -> Figure:
        if axes is None:
            fig, (a0, a1) = plt.subplots(2, 1, sharex=True, figsize=self.style.figsize, dpi=self.style.dpi,
                                         gridspec_kw={"height_ratios": [2, 1]})
        else:
            a0, a1 = axes
            fig = a0.figure
        x = self.dataset.time.plot_values()
        score, truth = _series(self.result, self.detector_id)
        a0.plot(x, self.dataset.value(self.parameter), color=self.style.observed_color, marker=".", ms=3, lw=1,
                label=self.parameter)
        for s, e in runs_of(truth == 1):
            a0.axvspan(x[s], x[e], color="#d62728", alpha=0.15, lw=0)
        for s, e in runs_of(truth == -1):
            a0.axvspan(x[s], x[e], color="0.5", alpha=0.15, lw=0, hatch="//")
        a0.legend(loc="upper left")
        a0.set_title(self.title)
        a1.plot(x, score, color="#1f77b4", lw=1.2, label="score")
        a1.axhline(self.threshold, color="#d62728", ls="--", label="alarm level")
        a1.set_ylabel("score")
        a1.legend(loc="upper left")
        a1.grid(alpha=0.3)
        fig.tight_layout()
        return fig


class ControlChart:
    """EWMA/CUSUM style chart: statistic against its control limit."""

    def __init__(self, dataset: EIIDataset, result: AnalysisResult, detector_id: str, style: ChartStyle | None = None):
        self.dataset, self.result, self.detector_id = dataset, result, detector_id
        self.style = style or ChartStyle(figsize=(14, 5))

    def render(self, ax: Any = None) -> Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        x = self.dataset.time.plot_values()
        score, truth = _series(self.result, self.detector_id)
        ax.plot(x, score, color="#1f77b4", lw=1.3, label="normalised statistic (|stat| / limit)")
        ax.axhline(1.0, color="#d62728", ls="--", label="control limit")
        ax.fill_between(x, 0, score, where=truth == 1, color="#d62728", alpha=0.25, label="out of control")
        ax.set_title(f"Control chart: {self.detector_id}")
        ax.set_ylabel("statistic / limit")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig


class ChangePointChart:
    """Signal with detected change points and segment means."""

    def __init__(self, dataset: EIIDataset, result: AnalysisResult, detector_id: str, parameter: str | None = None,
                 style: ChartStyle | None = None):
        self.dataset, self.result, self.detector_id = dataset, result, detector_id
        self.parameter = parameter or next((d["parameter"] for d in result.config.get("detectors", [])
                                            if d.get("detector_id") == detector_id), dataset.parameters[0])
        self.style = style or ChartStyle(figsize=(14, 5))

    def render(self, ax: Any = None) -> Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        x = self.dataset.time.plot_values()
        y = self.dataset.value(self.parameter)
        ax.plot(x, y, color=self.style.observed_color, lw=1, marker=".", ms=3, label=self.parameter)
        cps = [f for f in self.result.findings if f.detector_id == self.detector_id]
        bounds = [0] + sorted(f.start_index for f in cps) + [len(y)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = y[a:b]
            if np.any(~np.isnan(seg)):
                ax.hlines(np.nanmean(seg), x[a], x[b - 1], colors="#1f77b4", lw=2.5)
        for f in cps:
            ax.axvline(x[f.start_index], color="#d62728", ls="--", lw=1.2)
        ax.set_title(f"Change points: {self.detector_id}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig

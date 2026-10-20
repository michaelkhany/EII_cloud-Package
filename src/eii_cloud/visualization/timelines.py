"""Availability and event timelines."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..data import EIIDataset
from ..findings import AnalysisResult
from ..truth import runs_of
from .style import ChartStyle


class MissingnessTimeline:
    """One lane per parameter: observed (green), missing-when-expected (red), not expected (grey)."""

    def __init__(self, dataset: EIIDataset, result: AnalysisResult | None = None, parameters: list[str] | None = None,
                 style: ChartStyle | None = None):
        self.dataset, self.result = dataset, result
        self.parameters = parameters or dataset.parameters
        self.style = style or ChartStyle(figsize=(14, 0.7 * len(self.parameters) + 2))

    def render(self, ax: Any = None) -> Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        x = self.dataset.time.plot_values()
        for k, p in enumerate(self.parameters):
            obs = self.dataset.observed(p)
            exp = self.dataset.expected_mask(p)
            for mask, color, label in ((obs, "#2ca02c", "observed"), ((~obs) & exp, "#d62728", "missing when expected"),
                                       (~exp, "#bdbdbd", "not expected")):
                for s, e in runs_of(mask):
                    x0, x1 = x[s], x[e] if e > s else x[s]
                    bar = ax.barh(k, (x1 - x0) if e > s else (x[1] - x[0] if len(x) > 1 else 1), left=x0, height=0.6,
                                  color=color, label=label if k == 0 and s == runs_of(mask)[0][0] else None)
                    bar[0].set_gid(f"avail-{p}-{label.replace(' ', '_')}-{s}-{e}")
        ax.set_yticks(range(len(self.parameters)))
        ax.set_yticklabels(self.parameters)
        ax.invert_yaxis()
        ax.set_xlabel("Time")
        ax.set_title("Observation availability")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        return fig


class EventSequenceChart:
    """Findings and discrete events on a shared time axis (sequence rules)."""

    def __init__(self, dataset: EIIDataset, result: AnalysisResult, style: ChartStyle | None = None):
        self.dataset, self.result = dataset, result
        self.style = style or ChartStyle(figsize=(14, 6))

    def render(self, ax: Any = None) -> Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        x = self.dataset.time.plot_values()
        lanes: list[str] = []
        for f in self.result.findings:
            lane = f.rule_id or f.detector_id
            if lane not in lanes:
                lanes.append(lane)
            k = lanes.index(lane)
            color = self.style.flag_colors.get(f.category.value, "k")
            width = x[f.end_index] - x[f.start_index]
            if width == 0:
                width = (x[1] - x[0]) if len(x) > 1 else 1
            bar = ax.barh(k, width, left=x[f.start_index], height=0.5, color=color, alpha=0.5 if f.provisional else 0.9,
                          edgecolor="k", lw=0.5)
            bar[0].set_gid(f"seq-{f.finding_id}")
        ev = self.dataset.events.frame
        if len(ev):
            lanes.append("events")
            k = len(lanes) - 1
            for _, row in ev.iterrows():
                te = row["timestamp"]
                ax.plot([te], [k], marker="|", ms=18, color="k", mew=2)
                ax.annotate(str(row["event_type"]), (te, k), xytext=(3, 8), textcoords="offset points", fontsize=8)
        ax.set_yticks(range(len(lanes)))
        ax.set_yticklabels(lanes)
        ax.invert_yaxis()
        ax.set_xlabel("Time")
        ax.set_title("Evidence sequence: detector/rule findings and events")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

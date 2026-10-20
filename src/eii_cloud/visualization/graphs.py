"""Evidence-graph rendering (layered tree layout, no external graph library)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..evidence.graph import EvidenceGraph
from .style import ChartStyle

STATUS_COLORS = {"TRUE": "#c7e9c0", "FALSE": "#fcbba1", "UNKNOWN": "#d9d9d9", "MIXED": "#fdd0a2"}


class EvidenceGraphChart:
    def __init__(self, graph: EvidenceGraph, style: ChartStyle | None = None, title: str | None = None):
        self.graph = graph
        self.style = style or ChartStyle(figsize=(12, 7))
        self.title = title or f"Evidence graph for finding {graph.finding_id}"

    def render(self, ax: Any = None) -> Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure
        by_depth: dict[int, list] = defaultdict(list)
        for n in self.graph.nodes:
            by_depth[n.depth].append(n)
        pos: dict[str, tuple[float, float]] = {}
        maxd = max(by_depth) if by_depth else 0
        for d, nodes in by_depth.items():
            for k, n in enumerate(nodes):
                pos[n.id] = ((k + 1) / (len(nodes) + 1), 1 - d / (maxd + 1))
        for a, b in self.graph.edges:
            (x0, y0), (x1, y1) = pos[a], pos[b]
            ax.annotate("", xy=(x1, y1 + 0.04), xytext=(x0, y0 - 0.04),
                        arrowprops={"arrowstyle": "->", "color": "0.4", "lw": 1})
        for n in self.graph.nodes:
            x, y = pos[n.id]
            t = ax.text(x, y, f"{n.label}\n[{n.status}]", ha="center", va="center", fontsize=9,
                        bbox={"boxstyle": "round,pad=0.4", "fc": STATUS_COLORS.get(n.status, "white"), "ec": "0.4"})
            t.set_gid(f"evnode-{self.graph.finding_id}-{n.id}")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.08)
        ax.axis("off")
        ax.set_title(self.title)
        fig.tight_layout()
        return fig

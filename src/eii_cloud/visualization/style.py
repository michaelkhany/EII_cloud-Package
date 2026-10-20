"""Chart style tokens (defaults follow the reference EII Cloud figure)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChartStyle:
    figsize: tuple[float, float] = (20, 10)
    dpi: int = 150
    font_size: float = 12.0
    title_size: float = 16.0
    annotation_size: float = 11.0
    cloud_color: str = "#1f77b4"
    cloud_alpha: float = 0.25
    center_color: str = "#1f77b4"
    center_width: float = 2.0
    observed_color: str = "#ff7f0e"
    observed_width: float = 1.8
    marker: str = "o"
    marker_size: float = 4.0
    missing_color: str = "#1f77b4"
    missing_alpha: float = 0.10
    unsupported_hatch: str = "//"
    grid_alpha: float = 0.3
    legend_loc: str = "upper left"
    annotation_box: dict[str, Any] = field(default_factory=lambda: {"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.75", "alpha": 0.92})
    arrow: dict[str, Any] = field(default_factory=lambda: {"arrowstyle": "->", "color": "black", "lw": 1.0, "shrinkA": 2, "shrinkB": 4})
    howto_box: dict[str, Any] = field(default_factory=lambda: {"boxstyle": "round,pad=0.5", "fc": "#e8eef5", "ec": "0.6", "alpha": 0.95})
    flag_markers: dict[str, str] = field(default_factory=lambda: {
        "value_deviation": "^", "missing_when_expected": "x", "abnormal_flatline": "_",
        "context_contradiction": "D", "temporal_instability": "v", "cross_signal_inconsistency": "P",
        "composite_evidence": "*"})
    flag_colors: dict[str, str] = field(default_factory=lambda: {
        "value_deviation": "#d62728", "missing_when_expected": "#7f7f7f", "abnormal_flatline": "#9467bd",
        "context_contradiction": "#2ca02c", "temporal_instability": "#8c564b", "cross_signal_inconsistency": "#e377c2",
        "composite_evidence": "#17becf"})
    severity_colors: dict[str, str] = field(default_factory=lambda: {
        "info": "#7f7f7f", "low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728", "critical": "#8b0000"})
    transparent: bool = False
    timezone: str | None = None

    def with_(self, **kw: Any) -> ChartStyle:
        d = dict(self.__dict__)
        d.update(kw)
        return ChartStyle(**d)

"""Visualization registry: name -> chart class (extensible by users)."""
from __future__ import annotations

from collections.abc import Callable

_CHARTS: dict[str, type] = {}


def register_chart(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _CHARTS[name] = cls
        return cls

    return deco


def get_chart(name: str) -> type:
    if name not in _CHARTS:
        raise KeyError(f"unknown chart {name!r}; registered: {sorted(_CHARTS)}")
    return _CHARTS[name]


def list_charts() -> list[str]:
    return sorted(_CHARTS)


def _register_builtins() -> None:
    from .control_charts import AnomalyScoreChart, ChangePointChart, ControlChart
    from .eii_chart import EIICloudChart
    from .graphs import EvidenceGraphChart
    from .multi import ContextComparisonChart, DetectorComparisonChart, MultiSignalChart
    from .timelines import EventSequenceChart, MissingnessTimeline

    for cls in (EIICloudChart, AnomalyScoreChart, ControlChart, ChangePointChart, MissingnessTimeline,
                MultiSignalChart, ContextComparisonChart, EventSequenceChart, DetectorComparisonChart):
        _CHARTS[cls.__name__] = cls
    _CHARTS["EvidenceGraph"] = EvidenceGraphChart
    _CHARTS["EvidenceGraphChart"] = EvidenceGraphChart


_register_builtins()

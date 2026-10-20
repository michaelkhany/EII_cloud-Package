"""Detector interface, registry and shared helpers."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..context import ContextDefinition
from ..data import EIIDataset
from ..envelopes import ValidityEnvelope
from ..findings import Category, Finding, Severity
from ..truth import TruthSeries

_REGISTRY: dict[str, type[BaseDetector]] = {}


def register_detector(name: str) -> Callable[[type[BaseDetector]], type[BaseDetector]]:
    """Class decorator registering a detector under ``name`` (usable from YAML specs)."""

    def deco(cls: type[BaseDetector]) -> type[BaseDetector]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"detector name {name!r} already registered by {_REGISTRY[name].__name__}")
        cls.registry_name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_detector(name: str) -> type[BaseDetector]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_detectors() -> list[str]:
    return sorted(_REGISTRY)


def build_detector(spec: Mapping[str, Any]) -> BaseDetector:
    spec = dict(spec)
    name = spec.pop("type")
    return get_detector(name)(**spec)


@dataclass
class DetectorOutput:
    """What a detector returns: a truth series, optional score and findings."""

    detector_id: str
    truth: TruthSeries
    findings: list[Finding] = field(default_factory=list)
    score: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDetector:
    """Common interface for all detectors.

    Subclasses implement :meth:`evaluate` returning a :class:`TruthSeries` (plus optional
    score) and may override :meth:`describe` / :meth:`evidence_for`.  Fitting on reference
    data is optional via :meth:`fit`.
    """

    registry_name: str = "base"
    category: Category = Category.VALUE_DEVIATION
    default_visualization: str = "EIICloudChart"
    causal: bool = True  # True if output at t depends only on samples <= t (streaming-safe)

    def __init__(self, parameter: str, detector_id: str | None = None, severity: Severity | str = Severity.MEDIUM,
                 min_duration: int = 1, **params: Any):
        self.parameter = parameter
        self.detector_id = detector_id or f"{self.registry_name}:{parameter}"
        self.severity = Severity(severity)
        self.min_duration = int(min_duration)
        self.params = params
        self.fitted_: dict[str, Any] = {}

    # ---------------------------------------------------------------- API
    def fit(self, reference: EIIDataset) -> BaseDetector:  # noqa: D401 - optional
        return self

    def evaluate(self, dataset: EIIDataset, envelopes: Mapping[str, ValidityEnvelope],
                 context: ContextDefinition | None) -> tuple[TruthSeries, np.ndarray | None]:
        raise NotImplementedError

    def describe(self, dataset: EIIDataset, start: int, end: int, evidence: dict[str, Any]) -> str:
        return f"{self.category.title}: {self.parameter} flagged by {self.detector_id}."

    def evidence_for(self, dataset: EIIDataset, start: int, end: int, score: np.ndarray | None) -> dict[str, Any]:
        x = dataset.value(self.parameter)[start:end + 1]
        ev: dict[str, Any] = {"parameter": self.parameter, "n_samples": int(end - start + 1)}
        if np.any(~np.isnan(x)):
            ev.update({"observed_min": float(np.nanmin(x)), "observed_max": float(np.nanmax(x)),
                       "observed_mean": float(np.nanmean(x))})
        if score is not None:
            s = score[start:end + 1]
            if np.any(~np.isnan(s)):
                ev["score_max"] = float(np.nanmax(s))
        return ev

    def anchor(self, dataset: EIIDataset, start: int, end: int, score: np.ndarray | None) -> tuple[int, float | None]:
        """Sample index/value used to anchor chart annotations (default: mid-interval)."""
        i = (start + end) // 2
        x = dataset.value(self.parameter)
        v = x[i]
        return i, (None if np.isnan(v) else float(v))

    def reference_id(self, envelopes: Mapping[str, ValidityEnvelope]) -> str | None:
        return None

    def config(self) -> dict[str, Any]:
        return {"type": self.registry_name, "parameter": self.parameter, "detector_id": self.detector_id,
                "severity": self.severity.value, "min_duration": self.min_duration, **self.params}

    # ---------------------------------------------------------- execution
    def detect(self, dataset: EIIDataset, envelopes: Mapping[str, ValidityEnvelope] | None = None,
               context: ContextDefinition | None = None) -> DetectorOutput:
        envelopes = envelopes or {}
        truth, score = self.evaluate(dataset, envelopes, context)
        findings = []
        for start, end in truth.runs():
            if end - start + 1 < self.min_duration:
                continue
            ev = self.evidence_for(dataset, start, end, score)
            ai, av = self.anchor(dataset, start, end, score)
            findings.append(Finding(
                asset_id=dataset.asset_id, parameter_ids=self.parameter_ids(), category=self.category,
                start_time=dataset.time.label(start), end_time=dataset.time.label(end),
                start_index=start, end_index=end, detector_id=self.detector_id, severity=self.severity,
                evidence=ev, context_snapshot=context_snapshot(dataset, context, start, end),
                reference_id=self.reference_id(envelopes), description=self.describe(dataset, start, end, ev),
                magnitude=ev.get("magnitude"), score=ev.get("score_max"), anchor_index=ai, anchor_value=av,
                provenance={"detector": self.config()},
            ))
        return DetectorOutput(self.detector_id, truth, findings, score, {"fraction_unknown": truth.fraction_unknown()})

    def parameter_ids(self) -> list[str]:
        return [self.parameter]


def context_snapshot(dataset: EIIDataset, context: ContextDefinition | None, start: int, end: int) -> dict[str, Any]:
    snap: dict[str, Any] = {}
    for c in dataset.context.columns:
        col = dataset.context[c].iloc[start:end + 1]
        if col.dtype.kind in "fiu":
            snap[c] = {"min": float(col.min()), "max": float(col.max()), "mean": float(col.mean())}
        else:
            vals = col.astype(str).value_counts()
            snap[c] = {"values": vals.to_dict()}
    if context is not None:
        labels = context.regime_labels(dataset)[start:end + 1]
        uniq, counts = np.unique(labels.astype(str), return_counts=True)
        snap["regime"] = dict(zip(uniq.tolist(), counts.tolist()))
    return snap

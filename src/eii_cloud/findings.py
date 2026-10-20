"""Standardised analysis output: :class:`Finding` and :class:`AnalysisResult`."""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ._version import SCHEMA_VERSION, __version__
from .evidence.provenance import _json_default, provenance_record, stable_id
from .truth import TruthStatus


class Category(str, Enum):
    """Contextual anomaly categories A1-A7 of the specification."""

    VALUE_DEVIATION = "value_deviation"            # A1
    MISSING = "missing_when_expected"              # A2
    FLATLINE = "abnormal_flatline"                 # A3
    CONTEXT_CONTRADICTION = "context_contradiction"  # A4
    TEMPORAL_INSTABILITY = "temporal_instability"  # A5
    CROSS_SIGNAL = "cross_signal_inconsistency"    # A6
    COMPOSITE = "composite_evidence"               # A7

    @property
    def code(self) -> str:
        return {
            "value_deviation": "A1",
            "missing_when_expected": "A2",
            "abnormal_flatline": "A3",
            "context_contradiction": "A4",
            "temporal_instability": "A5",
            "cross_signal_inconsistency": "A6",
            "composite_evidence": "A7",
        }[self.value]

    @property
    def title(self) -> str:
        return {
            "value_deviation": "Value deviation",
            "missing_when_expected": "Observability degradation",
            "abnormal_flatline": "Abnormal flatline",
            "context_contradiction": "Context contradiction",
            "temporal_instability": "Temporal instability",
            "cross_signal_inconsistency": "Cross-signal inconsistency",
            "composite_evidence": "Composite evidence anomaly",
        }[self.value]


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return ["info", "low", "medium", "high", "critical"].index(self.value)


@dataclass
class Finding:
    """One detected anomaly with the evidence needed to reproduce it.

    ``start_time``/``end_time`` are *labels* in the dataset's own time representation
    (ISO-8601 strings for datetime axes, numbers for numeric axes). ``start_index`` and
    ``end_index`` are inclusive sample positions on the dataset time axis.
    """

    asset_id: str
    parameter_ids: list[str]
    category: Category
    start_time: Any
    end_time: Any
    start_index: int
    end_index: int
    detector_id: str
    rule_id: str | None = None
    severity: Severity = Severity.MEDIUM
    truth_status: TruthStatus = TruthStatus.TRUE
    evidence: dict[str, Any] = field(default_factory=dict)
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    reference_id: str | None = None
    description: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    magnitude: float | None = None
    score: float | None = None
    anchor_index: int | None = None
    anchor_value: float | None = None
    provisional: bool = False
    finding_id: str = ""

    def __post_init__(self) -> None:
        self.category = Category(self.category)
        self.severity = Severity(self.severity)
        self.truth_status = TruthStatus(self.truth_status)
        self.parameter_ids = list(self.parameter_ids)
        if self.anchor_index is None:
            self.anchor_index = int(self.end_index)
        if not self.finding_id:
            self.finding_id = stable_id(
                self.asset_id, self.parameter_ids, self.category.value, self.detector_id,
                self.rule_id, self.start_time, self.end_time,
            )

    # ------------------------------------------------------------- helpers
    @property
    def n_samples(self) -> int:
        return int(self.end_index - self.start_index + 1)

    def overlaps(self, start_index: int, end_index: int) -> bool:
        return not (self.end_index < start_index or self.start_index > end_index)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["category_code"] = self.category.code
        d["severity"] = self.severity.value
        d["truth_status"] = self.truth_status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Finding:
        d = dict(d)
        d.pop("category_code", None)
        return cls(**d)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"Finding({self.category.code} {self.category.value} {self.parameter_ids} "
                f"[{self.start_time} .. {self.end_time}] by {self.rule_id or self.detector_id}, "
                f"severity={self.severity.value}, id={self.finding_id})")


@dataclass
class AnalysisResult:
    """Findings plus everything needed to reproduce them."""

    analysis_id: str
    asset_id: str
    findings: list[Finding] = field(default_factory=list)
    envelopes: dict[str, dict[str, Any]] = field(default_factory=dict)
    truth_series: dict[str, list[int]] = field(default_factory=dict)
    scores: dict[str, list[float]] = field(default_factory=dict)
    dataset_manifest: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=provenance_record)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    library_version: str = __version__

    def __iter__(self) -> Iterator[Finding]:
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    def by_category(self, category: Category | str) -> list[Finding]:
        cat = Category(category)
        return [f for f in self.findings if f.category == cat]

    def by_parameter(self, parameter: str) -> list[Finding]:
        return [f for f in self.findings if parameter in f.parameter_ids]

    def final(self) -> list[Finding]:
        return [f for f in self.findings if not f.provisional]

    def sorted(self) -> AnalysisResult:
        self.findings.sort(key=lambda f: (f.start_index, f.end_index, f.category.value, f.detector_id, f.rule_id or ""))
        return self

    def get(self, finding_id: str) -> Finding:
        for f in self.findings:
            if f.finding_id == finding_id:
                return f
        raise KeyError(finding_id)

    def to_dict(self, include_series: bool = True) -> dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "library_version": self.library_version,
            "analysis_id": self.analysis_id,
            "asset_id": self.asset_id,
            "n_findings": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "envelopes": self.envelopes,
            "dataset_manifest": self.dataset_manifest,
            "config": self.config,
            "provenance": self.provenance,
            "warnings": self.warnings,
        }
        if include_series:
            d["truth_series"] = self.truth_series
            d["scores"] = self.scores
        return d

    def to_json(self, path: str | Path | None = None, include_series: bool = True, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(include_series), indent=indent, default=_json_default)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnalysisResult:
        return cls(
            analysis_id=d["analysis_id"],
            asset_id=d["asset_id"],
            findings=[Finding.from_dict(f) for f in d.get("findings", [])],
            envelopes=d.get("envelopes", {}),
            truth_series=d.get("truth_series", {}),
            scores=d.get("scores", {}),
            dataset_manifest=d.get("dataset_manifest", {}),
            config=d.get("config", {}),
            provenance=d.get("provenance", {}),
            warnings=d.get("warnings", []),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            library_version=d.get("library_version", __version__),
        )

    @classmethod
    def from_json(cls, path_or_text: str | Path) -> AnalysisResult:
        p = Path(path_or_text) if not str(path_or_text).lstrip().startswith("{") else None
        text = p.read_text() if p is not None and p.exists() else str(path_or_text)
        return cls.from_dict(json.loads(text))

    def summary(self) -> str:
        lines = [f"AnalysisResult {self.analysis_id} asset={self.asset_id}: {len(self.findings)} findings"]
        for f in self.findings:
            flag = " (provisional)" if f.provisional else ""
            lines.append(f"  - {f.category.code} {f.category.value:<28} {str(f.parameter_ids):<24} "
                         f"{f.start_time} .. {f.end_time}  by {f.rule_id or f.detector_id} [{f.severity.value}]{flag}")
        return "\n".join(lines)


def findings_equal(a: Iterable[Finding], b: Iterable[Finding]) -> bool:
    """Compare two finding collections by their stable identifiers."""
    return sorted(f.finding_id for f in a) == sorted(f.finding_id for f in b)

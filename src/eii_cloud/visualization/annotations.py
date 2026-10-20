"""Annotation contract: structured, renderer-independent annotation records.

:func:`build_annotations` is a pure function of an :class:`AnalysisResult` (and the dataset
for anchor coordinates).  It never touches Matplotlib, so the ``annotations_json`` rendering
mode and the image renderers share exactly the same records and identifiers.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .._version import SCHEMA_VERSION, __version__
from ..data import EIIDataset
from ..evidence.provenance import _json_default
from ..findings import AnalysisResult, Finding

ANNOTATION_SCHEMA_VERSION = "1.0"


@dataclass
class Annotation:
    annotation_id: str
    finding_id: str
    analysis_id: str
    asset_id: str
    parameter_ids: list[str]
    category: str
    category_code: str
    severity: str
    truth_status: str
    start_time: Any
    end_time: Any
    anchor_time: Any
    anchor_value: float | None
    text: str
    short_label: str
    rule_id: str | None
    detector_id: str
    reference_id: str | None
    evidence: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    envelope: dict[str, Any] | None = None
    versions: dict[str, Any] = field(default_factory=dict)
    provisional: bool = False
    number: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def short_label(finding: Finding) -> str:
    return {
        "value_deviation": "Deviation",
        "missing_when_expected": "Missing",
        "abnormal_flatline": "Flatline",
        "context_contradiction": "Contradiction",
        "temporal_instability": "Instability",
        "cross_signal_inconsistency": "Inconsistency",
        "composite_evidence": "Composite",
    }[finding.category.value]


def annotation_for(finding: Finding, result: AnalysisResult, dataset: EIIDataset | None, number: int) -> Annotation:
    anchor_i = finding.anchor_index if finding.anchor_index is not None else finding.end_index
    anchor_t = dataset.time.label(anchor_i) if dataset is not None else None
    anchor_v = finding.anchor_value
    if anchor_v is None and dataset is not None and finding.parameter_ids:
        p = finding.parameter_ids[0]
        if dataset.has_parameter(p):
            x = dataset.value(p)[finding.start_index:finding.end_index + 1]
            if np.any(~np.isnan(x)):
                anchor_v = float(np.nanmean(x))
    env_meta = None
    if finding.reference_id is not None:
        for e in result.envelopes.values():
            if e.get("reference_id") == finding.reference_id:
                env_meta = {k: v for k, v in e.items() if k not in ("lower", "upper", "center", "support")}
                break
    return Annotation(
        annotation_id=f"ann-{finding.finding_id}",
        finding_id=finding.finding_id,
        analysis_id=result.analysis_id,
        asset_id=finding.asset_id,
        parameter_ids=list(finding.parameter_ids),
        category=finding.category.value,
        category_code=finding.category.code,
        severity=finding.severity.value,
        truth_status=finding.truth_status.value,
        start_time=finding.start_time,
        end_time=finding.end_time,
        anchor_time=anchor_t,
        anchor_value=anchor_v,
        text=finding.description,
        short_label=short_label(finding),
        rule_id=finding.rule_id,
        detector_id=finding.detector_id,
        reference_id=finding.reference_id,
        evidence=_compact_evidence(finding.evidence),
        context=finding.context_snapshot,
        envelope=env_meta,
        versions={"library": __version__, "schema": SCHEMA_VERSION, "annotation_schema": ANNOTATION_SCHEMA_VERSION,
                  "rule_version": finding.provenance.get("rule_version"),
                  "detector": (finding.provenance.get("detector") or {}).get("type")},
        provisional=finding.provisional,
        number=number,
    )


def _compact_evidence(ev: dict[str, Any]) -> dict[str, Any]:
    """Evidence without the (large) nested explanation tree, but keeping the matched ids."""
    out = {k: v for k, v in ev.items() if k != "expression"}
    if "expression" in ev:
        out["expression_kind"] = ev["expression"].get("kind")
    return out


def build_annotations(result: AnalysisResult, dataset: EIIDataset | None = None, parameter: str | None = None,
                      categories: list[str] | None = None, include_provisional: bool = True) -> list[Annotation]:
    """Annotation records for (a subset of) the findings, numbered in time order."""
    finds = [f for f in result.findings
             if (parameter is None or parameter in f.parameter_ids)
             and (categories is None or f.category.value in categories)
             and (include_provisional or not f.provisional)]
    finds.sort(key=lambda f: (f.start_index, f.end_index, f.category.value))
    return [annotation_for(f, result, dataset, i + 1) for i, f in enumerate(finds)]


def annotations_document(result: AnalysisResult, dataset: EIIDataset | None = None, parameter: str | None = None,
                         **kw: Any) -> dict[str, Any]:
    anns = build_annotations(result, dataset, parameter, **kw)
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "library_version": __version__,
        "analysis_id": result.analysis_id,
        "asset_id": result.asset_id,
        "parameter": parameter,
        "n_annotations": len(anns),
        "annotations": [a.to_dict() for a in anns],
        "envelopes": {k: {kk: vv for kk, vv in v.items() if kk not in ("lower", "upper", "center", "support")}
                      for k, v in result.envelopes.items()},
        "missing_regions": [
            {"finding_id": f.finding_id, "parameter_ids": f.parameter_ids, "start_time": f.start_time, "end_time": f.end_time}
            for f in result.findings if f.category.value == "missing_when_expected"
            and (parameter is None or parameter in f.parameter_ids)
        ],
        "provenance": result.provenance,
    }


def write_annotations_json(result: AnalysisResult, path: str | Path, dataset: EIIDataset | None = None,
                           parameter: str | None = None, **kw: Any) -> Path:
    doc = annotations_document(result, dataset, parameter, **kw)
    p = Path(path)
    p.write_text(json.dumps(doc, indent=2, default=_json_default))
    return p

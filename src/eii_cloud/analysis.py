"""Analysis orchestration: :class:`EIIEngine` (batch) and :class:`StreamingEngine` (replay)."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .context import ContextDefinition
from .data import EIIDataset
from .detectors.base import BaseDetector, DetectorOutput, build_detector
from .envelopes import ValidityEnvelope, build_envelope
from .evidence.provenance import config_hash, provenance_record, stable_id
from .findings import AnalysisResult, Finding
from .rules import EvalContext, RuleSet


class EIIEngine:
    """Run envelopes, detectors and rules over a dataset and return an :class:`AnalysisResult`.

    Envelopes may be given as :class:`ValidityEnvelope` objects (bound to one dataset) or as
    serialisable specs (``{"method": ..., "parameter": ...}``) that are rebuilt for every
    dataset passed to :meth:`run`, which is what makes the engine reusable in streaming
    replay and in reproducibility manifests.
    """

    def __init__(self, envelopes: Mapping[str, ValidityEnvelope | Mapping[str, Any]] | None = None,
                 detectors: Iterable[BaseDetector | Mapping[str, Any]] = (), rules: RuleSet | None = None,
                 context: ContextDefinition | Mapping[str, Any] | None = None, event_tolerance: Any = None,
                 name: str = "eii"):
        self.envelope_specs: dict[str, ValidityEnvelope | dict[str, Any]] = {}
        for k, v in (envelopes or {}).items():
            self.add_envelope(k, v)
        self.detectors: list[BaseDetector] = []
        for d in detectors:
            self.add_detector(d)
        self.rules = rules if rules is not None else RuleSet()
        self.context = ContextDefinition.from_dict(context) if isinstance(context, Mapping) else context
        self.event_tolerance = event_tolerance
        self.name = name

    # ------------------------------------------------------------- config
    def add_envelope(self, name: str, envelope: ValidityEnvelope | Mapping[str, Any]) -> EIIEngine:
        if isinstance(envelope, ValidityEnvelope):
            envelope.name = name
            self.envelope_specs[name] = envelope
        else:
            self.envelope_specs[name] = dict(envelope)
        return self

    def add_detector(self, detector: BaseDetector | Mapping[str, Any]) -> EIIEngine:
        det = detector if isinstance(detector, BaseDetector) else build_detector(detector)
        if any(d.detector_id == det.detector_id for d in self.detectors):
            raise ValueError(f"duplicate detector_id {det.detector_id!r}")
        self.detectors.append(det)
        return self

    def add_rule(self, rule: Any) -> EIIEngine:
        self.rules.add(rule)
        return self

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "envelopes": {k: (v.to_dict() if isinstance(v, ValidityEnvelope) else v) for k, v in self.envelope_specs.items()},
            "detectors": [d.config() for d in self.detectors],
            "rules": self.rules.to_dict(),
            "context": self.context.to_dict() if self.context is not None else None,
            "event_tolerance": self.event_tolerance,
        }

    @property
    def causal(self) -> bool:
        return all(d.causal for d in self.detectors) and all(
            not isinstance(v, ValidityEnvelope) and v.get("causal", True) for v in self.envelope_specs.values())

    # -------------------------------------------------------------- run
    def resolve_envelopes(self, dataset: EIIDataset, reference: EIIDataset | None = None) -> dict[str, ValidityEnvelope]:
        out: dict[str, ValidityEnvelope] = {}
        for name, spec in self.envelope_specs.items():
            if isinstance(spec, ValidityEnvelope):
                if spec.n != dataset.n:
                    raise ValueError(f"envelope {name!r} has {spec.n} samples but dataset has {dataset.n}")
                out[name] = spec
            else:
                env = build_envelope(dataset, spec, reference=reference, context_definition=self.context)
                env.name = name
                out[name] = env
        return out

    def run(self, dataset: EIIDataset, reference: EIIDataset | None = None, fit: bool = True) -> AnalysisResult:
        warnings: list[str] = []
        envelopes = self.resolve_envelopes(dataset, reference)
        for env in envelopes.values():
            warnings.extend(f"envelope {env.name}: {w}" for w in env.warnings)
        outputs: dict[str, DetectorOutput] = {}
        for det in self.detectors:
            if reference is not None and fit:
                det.fit(reference)
            out = det.detect(dataset, envelopes, self.context)
            out.metadata.update({"category": det.category.value, "parameter": det.parameter, "causal": det.causal})
            outputs[det.detector_id] = out
        ctx = EvalContext(dataset=dataset, envelopes=envelopes, context=self.context, detector_outputs=outputs,
                          event_tolerance=self.event_tolerance)
        rule_results = self.rules.evaluate(ctx)
        findings: list[Finding] = [f for o in outputs.values() for f in o.findings]
        findings += [f for r in rule_results.values() for f in r.findings]
        cfg = self.config()
        result = AnalysisResult(
            analysis_id=stable_id(dataset.content_hash(), config_hash(cfg)),
            asset_id=dataset.asset_id,
            findings=findings,
            envelopes={k: v.to_dict(include_arrays=True) for k, v in envelopes.items()},
            truth_series={**{k: o.truth.values.tolist() for k, o in outputs.items()},
                          **{f"rule:{k}": r.truth.values.tolist() for k, r in rule_results.items()}},
            scores={k: _tolist(o.score) for k, o in outputs.items() if o.score is not None},
            dataset_manifest=dataset.manifest(),
            config=cfg,
            provenance=provenance_record(config_sha256=config_hash(cfg), reference=reference.manifest() if reference else None),
            warnings=warnings,
        )
        for env in envelopes.values():
            result.envelopes[env.name]["coverage"] = env.coverage(
                dataset.value(env.parameter), self.context.regime_labels(dataset) if self.context else None)
        result.provenance["lookahead"] = {k: r.lookahead for k, r in rule_results.items()}
        return result.sorted()


def _tolist(a: np.ndarray | None) -> list[float | None]:
    if a is None:
        return []
    return [None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in a.tolist()]


class StreamingEngine:
    """Event-time replay of an :class:`EIIEngine` with watermark-based finalisation.

    Chunks are appended to an internal buffer; after each ``push`` the batch engine is
    re-run over the buffer (exact, no approximation) and findings are classified as
    *final* when their interval has ended and the watermark has passed
    ``end_time + lookahead`` of the producing rule/detector, or *provisional* otherwise.
    Late events can be appended with :meth:`push_events`.  With causal detectors the
    finalised findings are identical to a batch run (EXP-06).
    """

    def __init__(self, engine: EIIEngine, reference: EIIDataset | None = None, allowed_lateness: Any = 0):
        self.engine = engine
        self.reference = reference
        self.allowed_lateness = allowed_lateness
        self._buffer: EIIDataset | None = None
        self.final: dict[str, Finding] = {}
        self.provisional: dict[str, Finding] = {}
        self.watermark: float | None = None
        if not engine.causal:
            import warnings as _w
            _w.warn("StreamingEngine: engine contains non-causal detectors/envelopes; "
                    "finalised findings may differ from batch results", RuntimeWarning, stacklevel=2)

    @property
    def buffer(self) -> EIIDataset | None:
        return self._buffer

    def push(self, chunk: EIIDataset, watermark: Any = None) -> AnalysisResult:
        if self._buffer is None:
            self._buffer = chunk
        else:
            if chunk.time.t[0] <= self._buffer.time.t[-1]:
                raise ValueError("chunk must start after the current buffer (out-of-order chunk)")
            self._buffer = _concat(self._buffer, chunk)
        wm = self._buffer.time.t[-1] if watermark is None else self._buffer.time.from_label(watermark)
        self.watermark = wm - self._buffer.time.offset(self.allowed_lateness)
        return self._evaluate()

    def push_events(self, events: Any) -> AnalysisResult:
        """Append late-arriving events to the buffer's event log and re-evaluate."""
        from .evidence.events import EventLog

        if self._buffer is None:
            raise RuntimeError("push data before events")
        new = events if isinstance(events, EventLog) else EventLog.from_records(events)
        merged = pd.concat([self._buffer.events.frame, new.frame], ignore_index=True)
        cov = pd.concat([self._buffer.events.coverage, new.coverage], ignore_index=True)
        self._buffer.events = EventLog(merged, cov)
        return self._evaluate()

    def flush(self) -> AnalysisResult:
        """Close the stream: everything becomes final."""
        self.watermark = float("inf")
        return self._evaluate()

    def _evaluate(self) -> AnalysisResult:
        assert self._buffer is not None
        result = self.engine.run(self._buffer, self.reference)
        t = self._buffer.time.t
        n = self._buffer.n
        lookahead = result.provenance.get("lookahead", {})
        det_causal = {d.detector_id: d.causal for d in self.engine.detectors}
        self.provisional = {}
        for f in result.findings:
            if f.rule_id is not None:
                la = lookahead.get(f.rule_id, 0.0)
            else:
                la = 0.0 if det_causal.get(f.detector_id, True) else float("inf")
            run_closed = f.end_index < n - 1
            final = self.watermark == float("inf") or (run_closed and t[f.end_index] + la < self.watermark)
            f.provisional = not final
            if final:
                self.final[f.finding_id] = f
            else:
                self.provisional[f.finding_id] = f
        result.findings = sorted(list(self.final.values()) + list(self.provisional.values()),
                                 key=lambda f: (f.start_index, f.end_index, f.category.value, f.detector_id, f.rule_id or ""))
        return result


def _concat(a: EIIDataset, b: EIIDataset) -> EIIDataset:
    from .evidence.events import EventLog

    time = a.time.index.append(b.time.index)
    values = pd.concat([a.values, b.values], ignore_index=True)
    context = pd.concat([a.context, b.context], ignore_index=True) if len(a.context.columns) else None
    expected = pd.concat([a.expected, b.expected], ignore_index=True)
    ev_frame = pd.concat([a.events.frame, b.events.frame], ignore_index=True).drop_duplicates("event_id")
    cov = pd.concat([a.events.coverage, b.events.coverage], ignore_index=True).drop_duplicates()
    return EIIDataset(time, values, context, expected, units=a.units, events=EventLog(ev_frame, cov),
                      asset_id=a.asset_id, name=a.name, attrs=a.attrs)


def split_chunks(dataset: EIIDataset, chunk_size: int) -> list[EIIDataset]:
    return [dataset.iloc(i, min(i + chunk_size, dataset.n)) for i in range(0, dataset.n, chunk_size)]

"""Evidence-condition nodes and temporal operators of the rule language.

Operator semantics (``w`` is a window in dataset time units; ``t`` a sample time):

================== ======================================================================
``A & B``           Kleene AND.  ``A | B``: Kleene OR.  ``~A``: Kleene NOT (UNKNOWN stays).
``A.before(B, w)``  TRUE at t iff A(t)=TRUE and B is TRUE at some s in ``[t, t+w]``.
``A.after(B, w)``   TRUE at t iff A(t)=TRUE and B is TRUE at some s in ``[t-w, t]``.
``A.within(B, w)``  TRUE at t iff A(t)=TRUE and B is TRUE at some s in ``[t-w, t+w]``.
``A.during(B)``     TRUE on a maximal TRUE run of A iff B is TRUE on the whole run.
``A.overlaps(B)``   TRUE on a maximal TRUE run of A iff B is TRUE somewhere in the run.
``A.persists_for(w)`` TRUE at t iff A is TRUE on every sample in ``[t-w, t]`` and that
                    window lies fully inside the observed axis.
``A.count_within(w, k)`` TRUE at t iff at least ``k`` onsets (rising edges) of A lie in
                    ``[t-w, t]``.
================== ======================================================================

For every windowed operator a window that reaches before the first or after the last
sample makes a non-TRUE result UNKNOWN rather than FALSE; in streaming mode such samples
stay *provisional* until the event-time watermark passes ``t + lookahead``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..context import ContextDefinition, compare
from ..data import EIIDataset
from ..detectors.base import DetectorOutput
from ..detectors.contextual import evaluate_expectation
from ..envelopes import ValidityEnvelope
from ..findings import Category
from ..truth import TruthSeries, kleene_all, kleene_any
from ._windows import run_level, window_all, window_any, window_count


class RuleError(Exception):
    """Base class for rule definition/evaluation errors."""


@dataclass
class EvalContext:
    """Everything a node may consult while evaluating."""

    dataset: EIIDataset
    envelopes: Mapping[str, ValidityEnvelope] = field(default_factory=dict)
    context: ContextDefinition | None = None
    detector_outputs: Mapping[str, DetectorOutput] = field(default_factory=dict)
    rule_results: dict[str, TruthSeries] = field(default_factory=dict)
    event_tolerance: float | str | None = None  # default: half the median sampling step

    @property
    def t(self) -> np.ndarray:
        return self.dataset.time.t

    def offset(self, w: Any) -> float:
        return self.dataset.time.offset(w)


class Node:
    """Base class of all rule-expression nodes."""

    kind: str = "node"

    def __init__(self) -> None:
        self.children: list[Node] = []

    # ------------------------------------------------------------- evaluate
    def evaluate(self, ctx: EvalContext) -> TruthSeries:
        raise NotImplementedError

    def explain(self, ctx: EvalContext, start: int, end: int) -> dict[str, Any]:
        ts = self.evaluate(ctx)
        seg = ts.values[start:end + 1]
        d: dict[str, Any] = {
            "kind": self.kind,
            "spec": self.to_dict(),
            "n_true": int(np.sum(seg == 1)),
            "n_false": int(np.sum(seg == 0)),
            "n_unknown": int(np.sum(seg == -1)),
        }
        d.update(self._evidence(ctx, start, end))
        if self.children:
            d["children"] = [c.explain(ctx, start, end) for c in self.children]
        return d

    def _evidence(self, ctx: EvalContext, start: int, end: int) -> dict[str, Any]:
        return {}

    # -------------------------------------------------------------- meta
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def lookahead(self, ctx: EvalContext) -> float:
        return max([c.lookahead(ctx) for c in self.children], default=0.0)

    def rule_refs(self) -> set[str]:
        s: set[str] = set()
        for c in self.children:
            s |= c.rule_refs()
        return s

    def parameters(self) -> list[str]:
        out: list[str] = []
        for c in self.children:
            for p in c.parameters():
                if p not in out:
                    out.append(p)
        return out

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    # ------------------------------------------------------- builder sugar
    def __and__(self, other: Node) -> And:
        return And([self, other])

    def __or__(self, other: Node) -> Or:
        return Or([self, other])

    def __invert__(self) -> Not:
        return Not(self)

    def before(self, other: Node, window: Any) -> Before:
        return Before(self, other, window)

    def after(self, other: Node, window: Any) -> After:
        return After(self, other, window)

    def within(self, other: Node, window: Any) -> Within:
        return Within(self, other, window)

    def during(self, other: Node) -> During:
        return During(self, other)

    def overlaps(self, other: Node) -> Overlaps:
        return Overlaps(self, other)

    def persists_for(self, window: Any) -> PersistsFor:
        return PersistsFor(self, window)

    def count_within(self, window: Any, min_count: int) -> CountWithin:
        return CountWithin(self, window, min_count)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.to_dict()})"


# ------------------------------------------------------------------ leaf conditions

class Signal(Node):
    kind = "signal"

    def __init__(self, signal: str, op: str, value: Any):
        super().__init__()
        self.signal, self.op, self.value = signal, op, value

    def evaluate(self, ctx):
        return compare(ctx.dataset.context_column(self.signal), self.op, self.value)

    def _evidence(self, ctx, start, end):
        col = np.asarray(ctx.dataset.context_column(self.signal))
        seg = col[start:end + 1]
        if seg.dtype.kind in "fiu":
            seg = seg.astype(float)
            return {"observed_min": float(np.nanmin(seg)) if np.any(~np.isnan(seg)) else None,
                    "observed_max": float(np.nanmax(seg)) if np.any(~np.isnan(seg)) else None}
        vals, counts = np.unique(seg.astype(str), return_counts=True)
        return {"observed_values": dict(zip(vals.tolist(), counts.tolist()))}

    def to_dict(self):
        return {"signal": self.signal, "op": self.op, "value": self.value}

    def parameters(self):
        return [self.signal]


class OutsideEnvelope(Node):
    kind = "outside_envelope"

    def __init__(self, parameter: str, envelope: str | None = None):
        super().__init__()
        self.parameter, self.envelope = parameter, envelope

    def _env(self, ctx: EvalContext) -> ValidityEnvelope:
        name = self.envelope
        if name is None:
            cands = [e for e in ctx.envelopes.values() if e.parameter == self.parameter]
            if len(cands) != 1:
                raise RuleError(f"outside_envelope({self.parameter}): specify 'envelope'; candidates: "
                                f"{[e.name for e in cands]}")
            return cands[0]
        if name not in ctx.envelopes:
            raise RuleError(f"envelope {name!r} not available; have {sorted(ctx.envelopes)}")
        return ctx.envelopes[name]

    def evaluate(self, ctx):
        env = self._env(ctx)
        return ~env.contains(ctx.dataset.value(self.parameter))

    def _evidence(self, ctx, start, end):
        env = self._env(ctx)
        x = ctx.dataset.value(self.parameter)[start:end + 1]
        exc = env.excess(ctx.dataset.value(self.parameter))[start:end + 1]
        ok = ~np.isnan(exc)
        return {"envelope": env.name, "reference_id": env.reference_id,
                "max_excess": float(np.nanmax(np.abs(exc))) if ok.any() else None,
                "observed_min": float(np.nanmin(x)) if np.any(~np.isnan(x)) else None,
                "observed_max": float(np.nanmax(x)) if np.any(~np.isnan(x)) else None}

    def to_dict(self):
        return {"outside_envelope": {"parameter": self.parameter, "envelope": self.envelope}}

    def parameters(self):
        return [self.parameter]


class MissingWhenExpected(Node):
    kind = "missing_when_expected"

    def __init__(self, parameter: str):
        super().__init__()
        self.parameter = parameter

    def evaluate(self, ctx):
        return TruthSeries.from_bool(ctx.dataset.missing_when_expected(self.parameter))

    def _evidence(self, ctx, start, end):
        return {"n_missing": int(ctx.dataset.missing_when_expected(self.parameter)[start:end + 1].sum())}

    def to_dict(self):
        return {"missing_when_expected": self.parameter}

    def parameters(self):
        return [self.parameter]


class Event(Node):
    """Discrete event mapped onto the sample grid.

    An event at time ``e`` marks the nearest sample whose time lies within ``tolerance``
    (default: half the median sampling step, i.e. nearest-sample matching).  Outside the
    source's declared coverage the result is UNKNOWN; sources with no coverage record are
    assumed fully covered.
    """

    kind = "event"

    def __init__(self, event_type: str, source: str | None = None, tolerance: Any = None):
        super().__init__()
        self.event_type, self.source, self.tolerance = event_type, source, tolerance
        self._matched: dict[int, list[str]] = {}

    def _tol(self, ctx: EvalContext) -> float:
        tol = self.tolerance if self.tolerance is not None else ctx.event_tolerance
        if tol is None:
            step = ctx.dataset.time.median_step
            return 0.5 * step if np.isfinite(step) else 0.0
        return ctx.offset(tol)

    def evaluate(self, ctx):
        ds = ctx.dataset
        t = ds.time.t
        n = len(t)
        present = np.zeros(n, dtype=bool)
        self._matched = {}
        events = ds.events.select(self.event_type, asset_id=ds.asset_id, source=self.source)
        tol = self._tol(ctx)
        for _, row in events.iterrows():
            te = ds.time.from_label(row["timestamp"])
            j = int(np.searchsorted(t, te))
            cands = [k for k in (j - 1, j) if 0 <= k < n]
            if not cands:
                continue
            k = min(cands, key=lambda k: abs(t[k] - te))
            if abs(t[k] - te) <= tol:
                present[k] = True
                self._matched.setdefault(k, []).append(str(row["event_id"]))
        unknown = np.zeros(n, dtype=bool)
        src = self.source or self.event_type
        cov = ds.events.coverage
        cov = cov[cov["source"].astype(str) == src] if len(cov) else cov
        if len(cov):
            covered = np.zeros(n, dtype=bool)
            for _, c in cov.iterrows():
                a, b = ds.time.from_label(c["start"]), ds.time.from_label(c["end"])
                covered |= (t >= a) & (t <= b)
            unknown = ~covered & ~present
        return TruthSeries.from_bool(present, unknown)

    def _evidence(self, ctx, start, end):
        self.evaluate(ctx)
        ids = [eid for k, lst in self._matched.items() if start <= k <= end for eid in lst]
        return {"matched_event_ids": ids, "event_type": self.event_type, "source": self.source or self.event_type}

    def to_dict(self):
        return {"event": {"event_type": self.event_type, "source": self.source, "tolerance": self.tolerance}}


class Context(Node):
    kind = "context"

    def __init__(self, regime: str):
        super().__init__()
        self.regime = regime

    def evaluate(self, ctx):
        if ctx.context is None:
            raise RuleError("context condition requires a ContextDefinition")
        return ctx.context.regime_series(ctx.dataset, self.regime)

    def to_dict(self):
        return {"context": self.regime}


class FindingPresent(Node):
    """TRUE where a detector output (by ``detector`` id or ``category``) is TRUE."""

    kind = "finding"

    def __init__(self, detector: str | None = None, category: str | None = None, parameter: str | None = None):
        super().__init__()
        if detector is None and category is None:
            raise RuleError("finding condition needs 'detector' or 'category'")
        self.detector, self.category, self.parameter = detector, category, parameter

    def _outputs(self, ctx: EvalContext) -> list[DetectorOutput]:
        outs = list(ctx.detector_outputs.values())
        if self.detector is not None:
            if self.detector not in ctx.detector_outputs:
                raise RuleError(f"detector {self.detector!r} output not available; have {sorted(ctx.detector_outputs)}")
            outs = [ctx.detector_outputs[self.detector]]
        if self.category is not None:
            cat = Category(self.category)
            outs = [o for o in outs if any(f.category == cat for f in o.findings) or
                    o.metadata.get("category") == cat.value]
        if self.parameter is not None:
            outs = [o for o in outs if o.metadata.get("parameter") == self.parameter or
                    any(self.parameter in f.parameter_ids for f in o.findings)]
        return outs

    def evaluate(self, ctx):
        outs = self._outputs(ctx)
        if not outs:
            return TruthSeries.constant(ctx.dataset.n, -1)
        return kleene_any([o.truth for o in outs])

    def _evidence(self, ctx, start, end):
        ids = [f.finding_id for o in self._outputs(ctx) for f in o.findings if f.overlaps(start, end)]
        return {"matched_finding_ids": ids}

    def to_dict(self):
        return {"finding": {"detector": self.detector, "category": self.category, "parameter": self.parameter}}

    def parameters(self):
        return [self.parameter] if self.parameter else []


class RuleRef(Node):
    kind = "rule"

    def __init__(self, rule_id: str):
        super().__init__()
        self.rule_id = rule_id

    def evaluate(self, ctx):
        if self.rule_id not in ctx.rule_results:
            raise RuleError(f"rule {self.rule_id!r} has not been evaluated (ordering/cycle problem)")
        return ctx.rule_results[self.rule_id]

    def to_dict(self):
        return {"rule": self.rule_id}

    def rule_refs(self):
        return {self.rule_id}


class ContradictsContext(Node):
    kind = "contradicts_context"

    def __init__(self, parameter: str, expectation: str | None = None):
        super().__init__()
        self.parameter, self.expectation = parameter, expectation

    def _exps(self, ctx: EvalContext):
        if ctx.context is None:
            raise RuleError("contradicts_context requires a ContextDefinition")
        exps = [e for e in ctx.context.expectations_for(self.parameter)
                if self.expectation is None or e.id == self.expectation]
        if not exps:
            raise RuleError(f"no expectations for {self.parameter!r}")
        return exps

    def evaluate(self, ctx):
        return kleene_any([evaluate_expectation(ctx.dataset, ctx.context, e) for e in self._exps(ctx)])

    def _evidence(self, ctx, start, end):
        return {"expectations": [e.id for e in self._exps(ctx)]}

    def to_dict(self):
        return {"contradicts_context": {"parameter": self.parameter, "expectation": self.expectation}}

    def parameters(self):
        return [self.parameter]


# --------------------------------------------------------------------- logical nodes

class And(Node):
    kind = "and"

    def __init__(self, children: list[Node]):
        super().__init__()
        if len(children) < 1:
            raise RuleError("and needs at least one child")
        self.children = list(children)

    def evaluate(self, ctx):
        return kleene_all([c.evaluate(ctx) for c in self.children])

    def to_dict(self):
        return {"and": [c.to_dict() for c in self.children]}


class Or(Node):
    kind = "or"

    def __init__(self, children: list[Node]):
        super().__init__()
        if len(children) < 1:
            raise RuleError("or needs at least one child")
        self.children = list(children)

    def evaluate(self, ctx):
        return kleene_any([c.evaluate(ctx) for c in self.children])

    def to_dict(self):
        return {"or": [c.to_dict() for c in self.children]}


class Not(Node):
    kind = "not"

    def __init__(self, child: Node):
        super().__init__()
        self.children = [child]

    def evaluate(self, ctx):
        return ~self.children[0].evaluate(ctx)

    def to_dict(self):
        return {"not": self.children[0].to_dict()}


# -------------------------------------------------------------------- temporal nodes

class _Windowed(Node):
    def __init__(self, a: Node, b: Node | None, window: Any):
        super().__init__()
        self.children = [a] if b is None else [a, b]
        self.window = window

    def _w(self, ctx: EvalContext) -> float:
        w = ctx.offset(self.window)
        if w < 0:
            raise RuleError("window must be non-negative")
        return w


class Before(_Windowed):
    kind = "before"

    def evaluate(self, ctx):
        a, b = self.children[0].evaluate(ctx), self.children[1].evaluate(ctx)
        return a & window_any(b, ctx.t, 0.0, self._w(ctx))

    def lookahead(self, ctx):
        return self._w(ctx) + super().lookahead(ctx)

    def to_dict(self):
        return {"before": {"condition": self.children[0].to_dict(), "next": self.children[1].to_dict(),
                           "window": self.window}}


class After(_Windowed):
    kind = "after"

    def evaluate(self, ctx):
        a, b = self.children[0].evaluate(ctx), self.children[1].evaluate(ctx)
        return a & window_any(b, ctx.t, self._w(ctx), 0.0)

    def to_dict(self):
        return {"after": {"condition": self.children[0].to_dict(), "prior": self.children[1].to_dict(),
                          "window": self.window}}


class Within(_Windowed):
    kind = "within"

    def evaluate(self, ctx):
        a, b = self.children[0].evaluate(ctx), self.children[1].evaluate(ctx)
        w = self._w(ctx)
        return a & window_any(b, ctx.t, w, w)

    def lookahead(self, ctx):
        return self._w(ctx) + super().lookahead(ctx)

    def to_dict(self):
        return {"within": {"condition": self.children[0].to_dict(), "other": self.children[1].to_dict(),
                           "window": self.window}}


class During(Node):
    kind = "during"

    def __init__(self, a: Node, b: Node):
        super().__init__()
        self.children = [a, b]

    def evaluate(self, ctx):
        return run_level(self.children[0].evaluate(ctx), self.children[1].evaluate(ctx), "during")

    def lookahead(self, ctx):
        return float("inf")  # needs the complete run of the first operand

    def to_dict(self):
        return {"during": {"condition": self.children[0].to_dict(), "interval": self.children[1].to_dict()}}


class Overlaps(Node):
    kind = "overlaps"

    def __init__(self, a: Node, b: Node):
        super().__init__()
        self.children = [a, b]

    def evaluate(self, ctx):
        return run_level(self.children[0].evaluate(ctx), self.children[1].evaluate(ctx), "overlaps")

    def lookahead(self, ctx):
        return float("inf")

    def to_dict(self):
        return {"overlaps": {"condition": self.children[0].to_dict(), "interval": self.children[1].to_dict()}}


class PersistsFor(_Windowed):
    kind = "persists_for"

    def __init__(self, a: Node, window: Any):
        super().__init__(a, None, window)

    def evaluate(self, ctx):
        return window_all(self.children[0].evaluate(ctx), ctx.t, self._w(ctx))

    def to_dict(self):
        return {"persists_for": {"condition": self.children[0].to_dict(), "window": self.window}}


class CountWithin(_Windowed):
    kind = "count_within"

    def __init__(self, a: Node, window: Any, min_count: int):
        super().__init__(a, None, window)
        self.min_count = int(min_count)

    def evaluate(self, ctx):
        ts, _ = window_count(self.children[0].evaluate(ctx), ctx.t, self._w(ctx), self.min_count)
        return ts

    def _evidence(self, ctx, start, end):
        _, count = window_count(self.children[0].evaluate(ctx), ctx.t, self._w(ctx), self.min_count)
        return {"max_count": int(count[start:end + 1].max())}

    def to_dict(self):
        return {"count_within": {"condition": self.children[0].to_dict(), "window": self.window,
                                 "min_count": self.min_count}}


class C:
    """Python-native builder: ``C.signal("gpu_util", ">=", 80) & C.event("gpu_xid")``."""

    signal = staticmethod(Signal)
    outside_envelope = staticmethod(OutsideEnvelope)
    missing_when_expected = staticmethod(MissingWhenExpected)
    event = staticmethod(Event)
    context = staticmethod(Context)
    finding = staticmethod(FindingPresent)
    rule = staticmethod(RuleRef)
    contradicts_context = staticmethod(ContradictsContext)
    all = staticmethod(And)
    any = staticmethod(Or)

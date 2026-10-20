"""Contextual detectors: context contradiction and cross-signal constraints."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..context import ContextDefinition, Expectation, regime_onset_mask
from ..data import EIIDataset
from ..findings import Category
from ..truth import TruthSeries, kleene_any
from .base import BaseDetector, register_detector


def evaluate_expectation(dataset: EIIDataset, context: ContextDefinition, exp: Expectation) -> TruthSeries:
    """TRUE where the expectation is *violated* while its regime is active (after ``settle``),
    FALSE where satisfied or the regime is inactive, UNKNOWN where the value or regime is missing."""
    regime = context.regime_series(dataset, exp.regime)
    x = dataset.value(exp.parameter)
    if exp.kind == "range":
        lo = -np.inf if exp.lower is None else exp.lower
        hi = np.inf if exp.upper is None else exp.upper
        unknown_v = np.isnan(x)
        viol = ~((np.where(unknown_v, 0, x) >= lo) & (np.where(unknown_v, 0, x) <= hi))
    elif exp.kind in ("min_level", "max_level"):
        unknown_v = np.isnan(x)
        viol = (np.where(unknown_v, 0, x) < exp.value) if exp.kind == "min_level" else (np.where(unknown_v, 0, x) > exp.value)
    else:  # variability over trailing window
        s = pd.Series(x).rolling(exp.window, min_periods=exp.window).std(ddof=0).to_numpy()
        unknown_v = np.isnan(s)
        safe = np.where(unknown_v, 0.0, s)
        viol = (safe < exp.value) if exp.kind == "min_variability" else (safe > exp.value)
    settle_mask = regime_onset_mask(regime.is_true, exp.settle)
    active = regime.is_true & ~settle_mask
    unknown = regime.is_unknown | (active & unknown_v)
    return TruthSeries.from_bool(active & viol & ~unknown_v, unknown)


@register_detector("context_contradiction")
class ContextContradictionDetector(BaseDetector):
    """A4 context contradiction: the parameter violates a regime-specific expectation.

    Uses the expectations declared for ``parameter`` in the engine's
    :class:`~eii_cloud.context.ContextDefinition` (or the subset named in ``expectations``).
    """

    category = Category.CONTEXT_CONTRADICTION
    default_visualization = "EIICloudChart"

    def __init__(self, parameter: str, expectations: list[str] | None = None, **kw: Any):
        super().__init__(parameter, expectations=expectations, **kw)
        self.expectation_ids = expectations
        self._active: list[Expectation] = []
        self._per: dict[str, TruthSeries] = {}

    def evaluate(self, dataset, envelopes, context):
        if context is None:
            raise ValueError("context_contradiction detector needs a ContextDefinition")
        exps = [e for e in context.expectations_for(self.parameter)
                if self.expectation_ids is None or e.id in self.expectation_ids]
        if not exps:
            raise ValueError(f"no contextual expectations defined for {self.parameter!r}")
        self._active = exps
        self._per = {e.id: evaluate_expectation(dataset, context, e) for e in exps}
        truth = kleene_any(list(self._per.values()))
        return truth, None

    def evidence_for(self, dataset, start, end, score):
        ev = super().evidence_for(dataset, start, end, score)
        ev["violated_expectations"] = [
            {"id": e.id, "regime": e.regime, "kind": e.kind, "value": e.value, "lower": e.lower, "upper": e.upper,
             "window": e.window, "settle": e.settle, "description": e.description}
            for e in self._active if np.any(self._per[e.id].is_true[start:end + 1])
        ]
        return ev

    def describe(self, dataset, start, end, evidence):
        viol = evidence.get("violated_expectations") or []
        regime = viol[0]["regime"] if viol else "this context"
        return (f"Context contradiction:\nUnder {regime.replace('_', ' ')},\nthe parameter is expected\nto respond differently.\n\n"
                f"Question answered:\nDoes it contradict context? Yes.")


@register_detector("cross_signal")
class CrossSignalConstraintDetector(BaseDetector):
    """A6 cross-signal inconsistency: ``lhs  op  rhs`` must hold between signals.

    ``lhs``/``rhs`` are linear expressions (``{"const": c, "signal": coef, ...}``) or a bare
    signal name; ``op`` is one of ``<=, <, >=, >, ==`` with absolute ``tolerance``.
    The finding lists every involved parameter.
    """

    category = Category.CROSS_SIGNAL
    default_visualization = "MultiSignalChart"

    def __init__(self, parameter: str, lhs: Mapping[str, float] | str, op: str, rhs: Mapping[str, float] | str,
                 tolerance: float = 0.0, **kw: Any):
        super().__init__(parameter, lhs=dict(lhs) if not isinstance(lhs, str) else lhs, op=op,
                         rhs=dict(rhs) if not isinstance(rhs, str) else rhs, tolerance=tolerance, **kw)
        self.lhs, self.op, self.rhs, self.tolerance = lhs, op, rhs, tolerance
        if op not in ("<=", "<", ">=", ">", "=="):
            raise ValueError("op must be one of <=, <, >=, >, ==")

    def _signals(self, expr: Any) -> list[str]:
        if isinstance(expr, str):
            return [expr]
        return [k for k in expr if k != "const"]

    def parameter_ids(self) -> list[str]:
        ids = [self.parameter] + self._signals(self.lhs) + self._signals(self.rhs)
        seen: list[str] = []
        for i in ids:
            if i not in seen:
                seen.append(i)
        return seen

    def _eval(self, dataset: EIIDataset, expr: Any) -> np.ndarray:
        if isinstance(expr, str):
            return np.asarray(dataset.context_column(expr), dtype=np.float64)
        out = np.zeros(dataset.n)
        for k, c in expr.items():
            out += float(c) if k == "const" else float(c) * np.asarray(dataset.context_column(k), dtype=np.float64)
        return out

    def evaluate(self, dataset, envelopes, context):
        a, b = self._eval(dataset, self.lhs), self._eval(dataset, self.rhs)
        resid = a - b
        unknown = np.isnan(resid)
        r = np.where(unknown, 0.0, resid)
        tol = self.tolerance
        ok = {"<=": r <= tol, "<": r < tol, ">=": r >= -tol, ">": r > -tol, "==": np.abs(r) <= tol}[self.op]
        return TruthSeries.from_bool(~ok, unknown), np.where(unknown, np.nan, resid)

    def evidence_for(self, dataset, start, end, score):
        ev = super().evidence_for(dataset, start, end, score)
        r = score[start:end + 1] if score is not None else np.array([np.nan])
        ev.update({"constraint": f"{self.lhs} {self.op} {self.rhs} (tol {self.tolerance})",
                   "residual_max": float(np.nanmax(np.abs(r))), "magnitude": float(np.nanmax(np.abs(r)))})
        return ev

    def describe(self, dataset, start, end, evidence):
        return (f"Cross-signal inconsistency:\nThe relationship\n{evidence.get('constraint')}\nis violated.\n\n"
                "Question answered:\nAre the signals mutually consistent? No.")

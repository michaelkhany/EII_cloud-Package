"""Operating context: regime labelling and contextual expectations."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import EIIDataset
from .truth import TruthSeries

_OPS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}

UNKNOWN_REGIME = "__unknown__"


def compare(values: np.ndarray, op: str, threshold: Any) -> TruthSeries:
    """Three-valued comparison of a (possibly NaN-containing) array with a threshold."""
    if op not in _OPS and op not in ("in", "not in"):
        raise ValueError(f"unsupported operator {op!r}; use one of {list(_OPS)} or 'in'/'not in'")
    values = np.asarray(values)
    if values.dtype.kind in "fciub":
        unknown = np.isnan(values.astype(float))
        safe = np.where(unknown, 0.0, values.astype(float))
        if op in ("in", "not in"):
            mask = np.isin(safe, list(threshold))
            if op == "not in":
                mask = ~mask
        else:
            mask = _OPS[op](safe, float(threshold))
        return TruthSeries.from_bool(mask, unknown)
    # categorical
    sval = np.array([None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in values], dtype=object)
    unknown = np.array([v is None or v == UNKNOWN_REGIME for v in sval])
    if op in ("in", "not in"):
        mask = np.array([v in {str(t) for t in threshold} for v in sval])
        if op == "not in":
            mask = ~mask
    elif op == "==":
        mask = sval == str(threshold)
    elif op == "!=":
        mask = sval != str(threshold)
    else:
        raise ValueError(f"operator {op!r} is not valid for categorical context")
    return TruthSeries.from_bool(mask, unknown)


@dataclass
class RegimeRule:
    """A regime is active where ``signal op value`` holds."""

    name: str
    signal: str
    op: str
    value: Any

    def evaluate(self, dataset: EIIDataset) -> TruthSeries:
        return compare(dataset.context_column(self.signal), self.op, self.value)


@dataclass
class Expectation:
    """Contextual behavioural expectation of a parameter in a regime.

    ``kind``:

    * ``range``: value must lie in ``[lower, upper]``.
    * ``min_variability``: rolling standard deviation over ``window`` samples must be at
      least ``value`` (e.g. temperature must respond to a high-load regime).
    * ``max_variability``: rolling standard deviation must not exceed ``value``.
    * ``min_level`` / ``max_level``: value must be at least/at most ``value``.

    ``settle`` is the number of samples after regime onset during which the expectation is
    not enforced (thermal inertia, permissible response delay).
    """

    regime: str
    parameter: str
    kind: str
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    window: int = 5
    settle: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        allowed = ("range", "min_variability", "max_variability", "min_level", "max_level")
        if self.kind not in allowed:
            raise ValueError(f"expectation kind must be one of {allowed}")
        if self.kind == "range" and (self.lower is None and self.upper is None):
            raise ValueError("range expectation needs lower and/or upper")
        if self.kind != "range" and self.value is None:
            raise ValueError(f"{self.kind} expectation needs 'value'")

    @property
    def id(self) -> str:
        return f"{self.regime}:{self.parameter}:{self.kind}"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ContextDefinition:
    """Named regimes derived from context columns plus contextual expectations."""

    regimes: list[RegimeRule] = field(default_factory=list)
    expectations: list[Expectation] = field(default_factory=list)
    default_regime: str = "default"
    regime_column: str | None = None

    # ----------------------------------------------------------------- build
    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> ContextDefinition:
        regimes = []
        for name, r in (spec.get("regimes") or {}).items():
            regimes.append(RegimeRule(name=name, signal=r["signal"], op=r.get("op", ">="), value=r["value"]))
        exps = [Expectation(**e) for e in spec.get("expectations", [])]
        return cls(regimes=regimes, expectations=exps, default_regime=spec.get("default_regime", "default"),
                   regime_column=spec.get("regime_column"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "regimes": {r.name: {"signal": r.signal, "op": r.op, "value": r.value} for r in self.regimes},
            "expectations": [e.to_dict() for e in self.expectations],
            "default_regime": self.default_regime,
            "regime_column": self.regime_column,
        }

    def add_regime(self, name: str, signal: str, op: str, value: Any) -> ContextDefinition:
        self.regimes.append(RegimeRule(name, signal, op, value))
        return self

    def expect(self, **kwargs: Any) -> ContextDefinition:
        self.expectations.append(Expectation(**kwargs))
        return self

    # ------------------------------------------------------------ evaluation
    def regime_labels(self, dataset: EIIDataset) -> np.ndarray:
        """Regime label per sample (first matching rule wins); ``__unknown__`` when
        the underlying context signal is missing."""
        n = dataset.n
        if self.regime_column is not None:
            col = dataset.context_column(self.regime_column)
            return np.array([UNKNOWN_REGIME if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                             for v in col], dtype=object)
        labels = np.full(n, self.default_regime, dtype=object)
        assigned = np.zeros(n, dtype=bool)
        unknown_any = np.zeros(n, dtype=bool)
        for rule in self.regimes:
            ts = rule.evaluate(dataset)
            hit = ts.is_true & ~assigned
            labels[hit] = rule.name
            assigned |= hit
            unknown_any |= ts.is_unknown
        labels[(~assigned) & unknown_any] = UNKNOWN_REGIME
        return labels

    def regime_series(self, dataset: EIIDataset, regime: str) -> TruthSeries:
        labels = self.regime_labels(dataset)
        return TruthSeries.from_bool(labels == regime, labels == UNKNOWN_REGIME)

    def regime_names(self) -> list[str]:
        return [r.name for r in self.regimes] + [self.default_regime]

    def expectations_for(self, parameter: str) -> list[Expectation]:
        return [e for e in self.expectations if e.parameter == parameter]


def regime_onset_mask(active: np.ndarray, settle: int) -> np.ndarray:
    """True for the first ``settle`` samples of each active run (where expectations are relaxed)."""
    active = np.asarray(active, dtype=bool)
    out = np.zeros_like(active)
    if settle <= 0:
        return out
    run = 0
    for i, a in enumerate(active):
        run = run + 1 if a else 0
        if a and run <= settle:
            out[i] = True
    return out

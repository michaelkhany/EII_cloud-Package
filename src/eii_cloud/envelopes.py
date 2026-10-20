"""EII Cloud validity envelopes: ``E_i(t | c) = [L_i(t,c), U_i(t,c)]`` with optional centre.

An envelope is **not** automatically a confidence or prediction interval. Every
:class:`ValidityEnvelope` records its construction method, reference population,
per-sample *support* (whether enough reference evidence existed to state a range) and
can report its empirical coverage.  Where support is missing the bounds are NaN and
downstream detectors return UNKNOWN rather than inventing a range.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .data import EIIDataset
from .evidence.provenance import stable_id
from .truth import TruthSeries

MAD_SCALE = 1.4826


@dataclass
class ValidityEnvelope:
    """Lower/centre/upper trajectories aligned with a dataset's time axis."""

    parameter: str
    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray | None = None
    support: np.ndarray | None = None
    method: str = "external_envelope"
    params: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)
        n = len(self.lower)
        if len(self.upper) != n:
            raise ValueError("lower/upper length mismatch")
        if self.center is not None:
            self.center = np.asarray(self.center, dtype=np.float64)
            if len(self.center) != n:
                raise ValueError("center length mismatch")
        if self.support is None:
            self.support = ~(np.isnan(self.lower) | np.isnan(self.upper))
        else:
            self.support = np.asarray(self.support, dtype=bool)
            if len(self.support) != n:
                raise ValueError("support length mismatch")
            self.support = self.support & ~(np.isnan(self.lower) | np.isnan(self.upper))
        bad = self.support & (self.lower > self.upper)
        if np.any(bad):
            raise ValueError(f"lower > upper at {int(bad.sum())} samples")
        if self.name is None:
            self.name = f"{self.parameter}:{self.method}"

    # ------------------------------------------------------------ identity
    @property
    def n(self) -> int:
        return len(self.lower)

    @property
    def reference_id(self) -> str:
        return stable_id(self.parameter, self.method, self.params, self.reference, self.name)

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    # ------------------------------------------------------------ evaluation
    def contains(self, values: np.ndarray) -> TruthSeries:
        """TRUE where observed value is inside, FALSE where outside, UNKNOWN where the
        value is missing or the envelope has no support."""
        values = np.asarray(values, dtype=np.float64)
        unknown = np.isnan(values) | ~self.support
        safe = np.where(np.isnan(values), 0.0, values)
        lo = np.where(self.support, self.lower, -np.inf)
        hi = np.where(self.support, self.upper, np.inf)
        inside = (safe >= lo) & (safe <= hi)
        return TruthSeries.from_bool(inside, unknown)

    def excess(self, values: np.ndarray) -> np.ndarray:
        """Signed distance outside the envelope (positive above, negative below, 0 inside)."""
        values = np.asarray(values, dtype=np.float64)
        above = np.clip(values - self.upper, 0, None)
        below = np.clip(self.lower - values, 0, None)
        out = np.where(above > 0, above, -below)
        return np.where(self.support, out, np.nan)

    def coverage(self, values: np.ndarray, by: np.ndarray | None = None) -> dict[str, Any]:
        """Empirical fraction of observed, supported samples inside the envelope."""
        values = np.asarray(values, dtype=np.float64)
        obs = ~np.isnan(values) & self.support
        inside = self.contains(values).is_true
        result: dict[str, Any] = {
            "n_supported_observed": int(obs.sum()),
            "n_unsupported": int((~self.support).sum()),
            "coverage": float(inside[obs].mean()) if obs.any() else float("nan"),
        }
        if by is not None:
            by = np.asarray(by, dtype=object)
            per: dict[str, Any] = {}
            for g in sorted({str(x) for x in by}):
                m = obs & (by.astype(str) == g)
                per[g] = {"n": int(m.sum()), "coverage": float(inside[m].mean()) if m.any() else float("nan")}
            result["by_context"] = per
        return result

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        d = {
            "name": self.name,
            "parameter": self.parameter,
            "method": self.method,
            "params": self.params,
            "reference": self.reference,
            "reference_id": self.reference_id,
            "n": self.n,
            "n_supported": int(self.support.sum()),
            "support_fraction": float(self.support.mean()),
            "warnings": self.warnings,
        }
        if include_arrays:
            d["lower"] = self.lower.tolist()
            d["upper"] = self.upper.tolist()
            d["center"] = None if self.center is None else self.center.tolist()
            d["support"] = self.support.tolist()
        return d

    def __repr__(self) -> str:  # pragma: no cover
        return f"ValidityEnvelope({self.name}, n={self.n}, supported={int(self.support.sum())}/{self.n})"


# ----------------------------------------------------------------------------- builders

def fixed_bounds(dataset: EIIDataset, parameter: str, lower: float, upper: float,
                 center: float | None = None, name: str | None = None) -> ValidityEnvelope:
    """Static engineering/operational thresholds."""
    n = dataset.n
    return ValidityEnvelope(
        parameter, np.full(n, lower), np.full(n, upper),
        None if center is None else np.full(n, center),
        method="fixed_bounds", params={"lower": lower, "upper": upper, "center": center}, name=name,
    )


def scheduled_bounds(dataset: EIIDataset, parameter: str, segments: Sequence[Mapping[str, Any]],
                     name: str | None = None) -> ValidityEnvelope:
    """Time-dependent limits.

    ``segments`` is a list of ``{"start": .., "end": .., "lower": .., "upper": .., "center": ..}``
    (``start``/``end`` optional, inclusive, in the dataset's time labels). Samples not
    covered by any segment have no support.
    """
    n = dataset.n
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    ce = np.full(n, np.nan)
    t = dataset.time.t
    for seg in segments:
        s = -np.inf if seg.get("start") is None else dataset.time.from_label(seg["start"])
        e = np.inf if seg.get("end") is None else dataset.time.from_label(seg["end"])
        m = (t >= s) & (t <= e)
        lo[m] = seg["lower"]
        hi[m] = seg["upper"]
        if seg.get("center") is not None:
            ce[m] = seg["center"]
    center = None if np.all(np.isnan(ce)) else ce
    return ValidityEnvelope(parameter, lo, hi, center, method="scheduled_bounds",
                            params={"segments": [dict(s) for s in segments]}, name=name)


def rolling_robust(dataset: EIIDataset, parameter: str, window: int = 20, k: float = 3.0,
                   min_periods: int | None = None, causal: bool = True, min_scale: float = 0.0,
                   name: str | None = None) -> ValidityEnvelope:
    """Rolling median +/- ``k`` * scaled MAD.

    With ``causal=True`` the window ends at the previous sample, so an observation never
    contributes to its own validity range (leakage-free, streaming-safe).
    """
    x = pd.Series(dataset.value(parameter))
    mp = max(3, window // 2) if min_periods is None else min_periods
    if causal:
        roll = x.shift(1).rolling(window, min_periods=mp)
    else:
        roll = x.rolling(window, min_periods=mp, center=True)
    med = roll.median()
    mad = roll.apply(lambda a: np.nanmedian(np.abs(a - np.nanmedian(a))), raw=True)
    scale = np.maximum(MAD_SCALE * mad.to_numpy(), min_scale)
    c = med.to_numpy()
    support = ~np.isnan(c) & ~np.isnan(scale)
    return ValidityEnvelope(parameter, c - k * scale, c + k * scale, c, support, method="rolling_robust",
                            params={"window": window, "k": k, "min_periods": mp, "causal": causal,
                                    "min_scale": min_scale},
                            reference={"population": "self (trailing)" if causal else "self (centred)"},
                            name=name)


def contextual_quantiles(dataset: EIIDataset, parameter: str, reference: EIIDataset, context: str,
                         q: tuple[float, float] = (0.05, 0.95), min_support: int = 30,
                         name: str | None = None) -> ValidityEnvelope:
    """Conditional empirical quantiles from a *reference* dataset, keyed by a context column.

    Contexts seen in ``dataset`` but with fewer than ``min_support`` reference observations
    receive no support (NaN bounds) and a warning is recorded instead of inventing a range.
    """
    ref_vals = reference.value(parameter)
    ref_ctx = np.asarray(reference.context_column(context)).astype(str)
    table: dict[str, dict[str, float]] = {}
    for g in np.unique(ref_ctx):
        v = ref_vals[(ref_ctx == g) & ~np.isnan(ref_vals)]
        if len(v) >= min_support:
            table[g] = {"lower": float(np.quantile(v, q[0])), "upper": float(np.quantile(v, q[1])),
                        "center": float(np.median(v)), "n": int(len(v))}
    ctx = np.asarray(dataset.context_column(context)).astype(str)
    n = dataset.n
    lo, hi, ce = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    warnings = []
    for g in np.unique(ctx):
        m = ctx == g
        if g in table:
            lo[m], hi[m], ce[m] = table[g]["lower"], table[g]["upper"], table[g]["center"]
        else:
            warnings.append(f"context {context}={g!r}: insufficient reference support (<{min_support}); no envelope")
    return ValidityEnvelope(parameter, lo, hi, ce, method="contextual_quantiles",
                            params={"context": context, "q": list(q), "min_support": min_support},
                            reference={"dataset": reference.name, "sha256": reference.content_hash(),
                                       "table": table}, name=name, warnings=warnings)


def regime_lookup(dataset: EIIDataset, parameter: str, table: Mapping[str, Mapping[str, float]],
                  context_definition: Any = None, context: str | None = None,
                  name: str | None = None) -> ValidityEnvelope:
    """Separate validity ranges for different operating regimes.

    ``table`` maps regime name -> ``{"lower", "upper", "center"?}``. Regimes come either
    from a :class:`~eii_cloud.context.ContextDefinition` or from a categorical context column.
    """
    if context_definition is not None:
        labels = context_definition.regime_labels(dataset)
    elif context is not None:
        labels = np.asarray(dataset.context_column(context)).astype(object)
    else:
        raise ValueError("regime_lookup needs context_definition or context")
    n = dataset.n
    lo, hi, ce = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    warnings = []
    for g in {str(x) for x in labels}:
        m = np.asarray([str(x) == g for x in labels])
        if g in table:
            lo[m], hi[m] = table[g]["lower"], table[g]["upper"]
            if table[g].get("center") is not None:
                ce[m] = table[g]["center"]
        else:
            warnings.append(f"regime {g!r} has no entry in lookup table; no envelope")
    center = None if np.all(np.isnan(ce)) else ce
    return ValidityEnvelope(parameter, lo, hi, center, method="regime_lookup",
                            params={"table": {k: dict(v) for k, v in table.items()}, "context": context},
                            name=name, warnings=warnings)


def regression_residual(dataset: EIIDataset, parameter: str, reference: EIIDataset, features: Sequence[str],
                        q: tuple[float, float] = (0.05, 0.95), model: Any = None, min_support: int = 30,
                        name: str | None = None) -> ValidityEnvelope:
    """Context-conditioned reference model with empirical residual limits.

    By default an ordinary least-squares linear model on ``features`` (numeric context or
    other parameters) is fitted on ``reference``; any object with ``fit(X, y)`` and
    ``predict(X)`` (e.g. scikit-learn estimators) can be supplied via ``model``.
    """
    def design(ds: EIIDataset) -> np.ndarray:
        cols = [np.asarray(ds.context_column(f), dtype=np.float64) for f in features]
        return np.column_stack(cols) if cols else np.zeros((ds.n, 0))

    Xr, yr = design(reference), reference.value(parameter)
    ok = ~np.isnan(yr) & ~np.isnan(Xr).any(axis=1)
    if ok.sum() < min_support:
        n = dataset.n
        return ValidityEnvelope(parameter, np.full(n, np.nan), np.full(n, np.nan), method="regression_residual",
                                params={"features": list(features), "q": list(q)},
                                warnings=[f"insufficient reference support ({int(ok.sum())} < {min_support})"], name=name)
    if model is None:
        A = np.column_stack([np.ones(ok.sum()), Xr[ok]])
        coef, *_ = np.linalg.lstsq(A, yr[ok], rcond=None)

        def predict(X: np.ndarray) -> np.ndarray:
            return np.column_stack([np.ones(len(X)), X]) @ coef

        model_desc: dict[str, Any] = {"type": "ols", "coef": coef.tolist()}
    else:
        model.fit(Xr[ok], yr[ok])
        predict = model.predict
        model_desc = {"type": type(model).__name__}
    resid = yr[ok] - predict(Xr[ok])
    rl, ru = float(np.quantile(resid, q[0])), float(np.quantile(resid, q[1]))
    X = design(dataset)
    valid = ~np.isnan(X).any(axis=1)
    c = np.full(dataset.n, np.nan)
    if valid.any():
        c[valid] = predict(X[valid])
    return ValidityEnvelope(parameter, c + rl, c + ru, c, valid, method="regression_residual",
                            params={"features": list(features), "q": list(q), "residual_quantiles": [rl, ru],
                                    "model": model_desc},
                            reference={"dataset": reference.name, "sha256": reference.content_hash(),
                                       "n_fit": int(ok.sum())}, name=name)


def physical_constraints(dataset: EIIDataset, parameter: str,
                         lower: Mapping[str, float] | Callable[[EIIDataset], np.ndarray] | None = None,
                         upper: Mapping[str, float] | Callable[[EIIDataset], np.ndarray] | None = None,
                         center: Mapping[str, float] | Callable[[EIIDataset], np.ndarray] | None = None,
                         name: str | None = None) -> ValidityEnvelope:
    """Bounds derived from documented engineering relationships.

    Each bound is either a *linear expression* mapping other parameter/context names to
    coefficients (``{"const": 30, "gpu_power": 0.1}``, serialisable and safe to load from
    YAML) or a Python callable of the dataset.
    """
    def evaluate(expr: Any) -> np.ndarray:
        if expr is None:
            return np.full(dataset.n, np.nan)
        if callable(expr):
            return np.asarray(expr(dataset), dtype=np.float64)
        out = np.zeros(dataset.n)
        for k, coef in expr.items():
            if k == "const":
                out += float(coef)
            else:
                out += float(coef) * np.asarray(dataset.context_column(k), dtype=np.float64)
        return out

    lo = evaluate(lower) if lower is not None else np.full(dataset.n, -np.inf)
    hi = evaluate(upper) if upper is not None else np.full(dataset.n, np.inf)
    ce = evaluate(center) if center is not None else None
    desc = {k: (v if not callable(v) else "<callable>") for k, v in
            {"lower": lower, "upper": upper, "center": center}.items()}
    return ValidityEnvelope(parameter, lo, hi, ce, method="physical_constraints", params=desc, name=name)


def external_envelope(dataset: EIIDataset, parameter: str, lower: Sequence[float], upper: Sequence[float],
                      center: Sequence[float] | None = None, times: Sequence[Any] | None = None,
                      name: str | None = None, source: str = "user") -> ValidityEnvelope:
    """User-supplied lower, centre and upper trajectories.

    Without ``times`` the arrays must have exactly one value per dataset sample.  With
    ``times`` (labels in the dataset's time representation) the trajectories are aligned by
    exact time match, so the same specification works for any slice or growing buffer of the
    data; samples with no supplied time have no support.
    """
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    center_arr = None if center is None else np.asarray(center, dtype=np.float64)
    if times is None:
        if len(lower) != dataset.n:
            raise ValueError("external envelope length must equal dataset length (or pass 'times')")
        lo, hi, ce = lower, upper, center_arr
    else:
        tt = np.array([dataset.time.from_label(x) for x in times], dtype=np.float64)
        if len(tt) != len(lower):
            raise ValueError("'times' must have the same length as the trajectories")
        order = np.argsort(tt)
        tt, lower, upper = tt[order], lower[order], upper[order]
        if center_arr is not None:
            center_arr = center_arr[order]
        t = dataset.time.t
        pos = np.searchsorted(tt, t)
        pos = np.clip(pos, 0, len(tt) - 1)
        tol = 1e-9 * max(1.0, float(np.max(np.abs(tt))))
        hit = np.abs(tt[pos] - t) <= tol
        lo, hi = np.full(dataset.n, np.nan), np.full(dataset.n, np.nan)
        lo[hit], hi[hit] = lower[pos[hit]], upper[pos[hit]]
        ce = None
        if center_arr is not None:
            ce = np.full(dataset.n, np.nan)
            ce[hit] = center_arr[pos[hit]]
    return ValidityEnvelope(parameter, lo, hi, ce, method="external_envelope",
                            reference={"source": source, "aligned_by_time": times is not None}, name=name)


class Forecaster(Protocol):
    """Plug-in protocol for :func:`forecast_interval`."""

    def fit(self, dataset: EIIDataset, parameter: str) -> Forecaster: ...

    def predict_interval(self, dataset: EIIDataset, parameter: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lower, center, upper) aligned with ``dataset`` (NaN where unavailable)."""
        ...


class SeasonalNaiveForecaster:
    """Reference implementation: value one ``period`` ago +/- ``k`` * residual scale.

    Strictly causal: the interval at ``t`` depends only on samples before ``t``.
    """

    def __init__(self, period: int, k: float = 3.0):
        self.period = period
        self.k = k
        self.scale_: float | None = None

    def fit(self, dataset: EIIDataset, parameter: str) -> SeasonalNaiveForecaster:
        x = dataset.value(parameter)
        if len(x) <= self.period:
            raise ValueError("reference shorter than period")
        resid = x[self.period:] - x[:-self.period]
        resid = resid[~np.isnan(resid)]
        self.scale_ = float(MAD_SCALE * np.median(np.abs(resid - np.median(resid)))) if len(resid) else float("nan")
        return self

    def predict_interval(self, dataset: EIIDataset, parameter: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.scale_ is None:
            raise RuntimeError("fit first")
        x = dataset.value(parameter)
        c = np.full_like(x, np.nan)
        c[self.period:] = x[:-self.period]
        return c - self.k * self.scale_, c, c + self.k * self.scale_


def forecast_interval(dataset: EIIDataset, parameter: str, forecaster: Forecaster,
                      reference: EIIDataset | None = None, name: str | None = None) -> ValidityEnvelope:
    """Prediction interval from a fitted forecaster (optional, predictive method)."""
    forecaster.fit(reference if reference is not None else dataset, parameter)
    lo, c, hi = forecaster.predict_interval(dataset, parameter)
    return ValidityEnvelope(parameter, lo, hi, c, method="forecast_interval",
                            params={"forecaster": type(forecaster).__name__, **{k: v for k, v in vars(forecaster).items()
                                                                                if not k.endswith("_")}},
                            reference={"dataset": (reference or dataset).name}, name=name)


ENVELOPE_METHODS: dict[str, Callable[..., ValidityEnvelope]] = {
    "fixed_bounds": fixed_bounds,
    "scheduled_bounds": scheduled_bounds,
    "rolling_robust": rolling_robust,
    "contextual_quantiles": contextual_quantiles,
    "regime_lookup": regime_lookup,
    "regression_residual": regression_residual,
    "physical_constraints": physical_constraints,
    "external_envelope": external_envelope,
    "forecast_interval": forecast_interval,
}


def build_envelope(dataset: EIIDataset, spec: Mapping[str, Any], reference: EIIDataset | None = None,
                   context_definition: Any = None) -> ValidityEnvelope:
    """Construct an envelope from a serialisable specification (e.g. loaded from YAML).

    ``spec`` = ``{"method": ..., "parameter": ..., "name": ..., **method_kwargs}``.
    """
    spec = dict(spec)
    method = spec.pop("method")
    if method not in ENVELOPE_METHODS:
        raise ValueError(f"unknown envelope method {method!r}; available: {sorted(ENVELOPE_METHODS)}")
    parameter = spec.pop("parameter")
    fn = ENVELOPE_METHODS[method]
    if method in ("contextual_quantiles", "regression_residual"):
        if reference is None:
            raise ValueError(f"{method} requires a reference dataset")
        spec["reference"] = reference
    if method == "regime_lookup" and context_definition is not None and "context" not in spec:
        spec["context_definition"] = context_definition
    if method == "forecast_interval":
        fc = spec.pop("forecaster", {"type": "seasonal_naive", "period": 24})
        if isinstance(fc, Mapping):
            if fc.get("type", "seasonal_naive") != "seasonal_naive":
                raise ValueError("only 'seasonal_naive' forecaster can be built from a spec; pass an object in Python")
            fc = SeasonalNaiveForecaster(period=int(fc["period"]), k=float(fc.get("k", 3.0)))
        spec["forecaster"] = fc
        spec["reference"] = reference
    if "q" in spec:
        spec["q"] = tuple(spec["q"])
    return fn(dataset, parameter, **spec)

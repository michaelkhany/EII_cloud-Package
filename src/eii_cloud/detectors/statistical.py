"""Statistical detectors: threshold/value deviation, robust deviation, EWMA, CUSUM."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..data import EIIDataset
from ..envelopes import MAD_SCALE, ValidityEnvelope, fixed_bounds
from ..findings import Category
from ..truth import TruthSeries
from .base import BaseDetector, register_detector


@register_detector("threshold")
class ThresholdDetector(BaseDetector):
    """A1 value deviation: observed value outside a validity envelope.

    ``envelope`` names an envelope in the engine's envelope map; alternatively static
    ``lower``/``upper`` limits build a ``fixed_bounds`` envelope on the fly.  Missing
    values and samples without envelope support yield UNKNOWN, not FALSE.
    """

    category = Category.VALUE_DEVIATION
    default_visualization = "EIICloudChart"

    def __init__(self, parameter: str, envelope: str | None = None, lower: float | None = None,
                 upper: float | None = None, **kw: Any):
        super().__init__(parameter, envelope=envelope, lower=lower, upper=upper, **kw)
        self.envelope_name = envelope
        self.lower, self.upper = lower, upper
        self._env: ValidityEnvelope | None = None

    def _resolve(self, dataset: EIIDataset, envelopes: Mapping[str, ValidityEnvelope]) -> ValidityEnvelope:
        if self.envelope_name is not None:
            if self.envelope_name not in envelopes:
                raise KeyError(f"envelope {self.envelope_name!r} not provided; available: {sorted(envelopes)}")
            return envelopes[self.envelope_name]
        if self.lower is None and self.upper is None:
            raise ValueError("threshold detector needs an envelope name or lower/upper limits")
        return fixed_bounds(dataset, self.parameter,
                            -np.inf if self.lower is None else self.lower,
                            np.inf if self.upper is None else self.upper)

    def evaluate(self, dataset, envelopes, context):
        self._env = env = self._resolve(dataset, envelopes)
        if env.parameter != self.parameter:
            raise ValueError(f"envelope {env.name} is for {env.parameter!r}, not {self.parameter!r}")
        x = dataset.value(self.parameter)
        inside = env.contains(x)
        return ~inside, env.excess(x)

    def reference_id(self, envelopes):
        return self._env.reference_id if self._env is not None else None

    def evidence_for(self, dataset, start, end, score):
        ev = super().evidence_for(dataset, start, end, score)
        env = self._env
        assert env is not None
        exc = score[start:end + 1] if score is not None else np.zeros(end - start + 1)
        j = int(np.nanargmax(np.abs(exc)))
        ev.update({
            "envelope": env.name, "envelope_method": env.method,
            "direction": "above" if np.nanmax(exc) > 0 else "below",
            "magnitude": float(np.nanmax(np.abs(exc))),
            "peak_index": int(start + j),
            "peak_value": float(dataset.value(self.parameter)[start + j]),
            "bound_at_peak": float(env.upper[start + j] if exc[j] > 0 else env.lower[start + j]),
        })
        return ev

    def anchor(self, dataset, start, end, score):
        if score is None:
            return super().anchor(dataset, start, end, score)
        j = int(np.nanargmax(np.abs(score[start:end + 1])))
        return start + j, float(dataset.value(self.parameter)[start + j])

    def describe(self, dataset, start, end, evidence):
        d = evidence.get("direction", "outside")
        verb = "rises above" if d == "above" else "falls below"
        return (f"{Category.VALUE_DEVIATION.title}:\nThe parameter is present,\nbut {verb} the EII Cloud.\n\n"
                f"Question answered:\nIs it inside its expected range? No.")


@register_detector("robust_deviation")
class RobustDeviationDetector(BaseDetector):
    """Rolling median/MAD z-score of the observed value versus a trailing window.

    Score ``|x - median| / (1.4826 MAD)`` (``NaN`` where the window is unsupported).
    """

    category = Category.VALUE_DEVIATION
    default_visualization = "AnomalyScoreChart"

    def __init__(self, parameter: str, window: int = 20, k: float = 4.0, min_periods: int | None = None,
                 min_scale: float = 1e-9, **kw: Any):
        super().__init__(parameter, window=window, k=k, min_periods=min_periods, min_scale=min_scale, **kw)
        self.window, self.k, self.min_scale = window, k, min_scale
        self.min_periods = max(3, window // 2) if min_periods is None else min_periods

    def evaluate(self, dataset, envelopes, context):
        x = pd.Series(dataset.value(self.parameter))
        roll = x.shift(1).rolling(self.window, min_periods=self.min_periods)
        med = roll.median().to_numpy()
        mad = roll.apply(lambda a: np.nanmedian(np.abs(a - np.nanmedian(a))), raw=True).to_numpy()
        scale = np.maximum(MAD_SCALE * mad, self.min_scale)
        score = np.abs(x.to_numpy() - med) / scale
        unknown = np.isnan(score)
        return TruthSeries.from_bool(np.where(unknown, False, score > self.k), unknown), score

    def describe(self, dataset, start, end, evidence):
        return (f"Robust deviation:\nThe value departs from its recent\nmedian by more than {self.k} robust s.d.\n\n"
                f"Question answered:\nIs it consistent with recent behaviour? No.")


@register_detector("ewma")
class EWMADetector(BaseDetector):
    """EWMA control chart around a reference level.

    The reference mean/sd come from :meth:`fit` (reference data) or, if absent, from an
    envelope centre (``envelope``) or the first ``burn_in`` samples.  The control limit is
    ``L * sigma * sqrt(lam / (2 - lam) * (1 - (1 - lam)^(2 i)))``.
    """

    category = Category.TEMPORAL_INSTABILITY
    default_visualization = "ControlChart"

    def __init__(self, parameter: str, lam: float = 0.2, L: float = 3.0, burn_in: int = 30,
                 envelope: str | None = None, **kw: Any):
        super().__init__(parameter, lam=lam, L=L, burn_in=burn_in, envelope=envelope, **kw)
        self.lam, self.L, self.burn_in, self.envelope_name = lam, L, burn_in, envelope
        self.limits_: np.ndarray | None = None
        self.center_: np.ndarray | None = None

    def fit(self, reference: EIIDataset):
        x = reference.value(self.parameter)
        x = x[~np.isnan(x)]
        self.fitted_ = {"mu": float(np.mean(x)), "sigma": float(np.std(x, ddof=1)), "n": int(len(x)),
                        "reference": reference.name}
        return self

    def evaluate(self, dataset, envelopes, context):
        x = dataset.value(self.parameter)
        n = len(x)
        if self.fitted_:
            mu = np.full(n, self.fitted_["mu"])
            sigma = self.fitted_["sigma"]
        elif self.envelope_name is not None:
            env = envelopes[self.envelope_name]
            if env.center is None:
                raise ValueError("EWMA with envelope needs an envelope centre")
            mu = env.center
            sigma = float(np.nanmedian(env.width) / 6.0)  # band ~ +/-3 sigma convention
        else:
            head = x[:self.burn_in]
            head = head[~np.isnan(head)]
            if len(head) < 3:
                raise ValueError("not enough burn-in samples for EWMA baseline")
            mu = np.full(n, float(np.mean(head)))
            sigma = float(np.std(head, ddof=1))
        z = np.full(n, np.nan)
        limit = np.full(n, np.nan)
        prev = 0.0  # z_0 = mu
        i_eff = 0
        for i in range(n):
            if np.isnan(x[i]) or np.isnan(mu[i]):
                continue
            prev = self.lam * (x[i] - mu[i]) + (1 - self.lam) * prev
            i_eff += 1
            z[i] = prev
            limit[i] = self.L * sigma * np.sqrt(self.lam / (2 - self.lam) * (1 - (1 - self.lam) ** (2 * i_eff)))
        self.limits_, self.center_ = limit, mu
        score = np.abs(z) / np.where(limit > 0, limit, np.nan)
        unknown = np.isnan(score)
        return TruthSeries.from_bool(np.where(unknown, False, score > 1.0), unknown), score

    def describe(self, dataset, start, end, evidence):
        return ("Process shift (EWMA):\nThe smoothed deviation from the\nreference level exceeds its control limit.\n\n"
                "Question answered:\nIs the process at its reference level? No.")


@register_detector("cusum")
class CUSUMDetector(BaseDetector):
    """Two-sided tabular CUSUM for sustained shifts of ``k`` sigma, alarm at ``h`` sigma."""

    category = Category.TEMPORAL_INSTABILITY
    default_visualization = "ControlChart"

    def __init__(self, parameter: str, k: float = 0.5, h: float = 5.0, burn_in: int = 30, reset: bool = True,
                 **kw: Any):
        super().__init__(parameter, k=k, h=h, burn_in=burn_in, reset=reset, **kw)
        self.k, self.h, self.burn_in, self.reset = k, h, burn_in, reset
        self.pos_: np.ndarray | None = None
        self.neg_: np.ndarray | None = None

    def fit(self, reference: EIIDataset):
        x = reference.value(self.parameter)
        x = x[~np.isnan(x)]
        self.fitted_ = {"mu": float(np.mean(x)), "sigma": float(np.std(x, ddof=1)), "n": int(len(x)),
                        "reference": reference.name}
        return self

    def evaluate(self, dataset, envelopes, context):
        x = dataset.value(self.parameter)
        if self.fitted_:
            mu, sigma = self.fitted_["mu"], self.fitted_["sigma"]
        else:
            head = x[:self.burn_in]
            head = head[~np.isnan(head)]
            if len(head) < 3:
                raise ValueError("not enough burn-in samples for CUSUM baseline")
            mu, sigma = float(np.mean(head)), float(np.std(head, ddof=1))
        sigma = sigma if sigma > 0 else 1e-9
        n = len(x)
        sp = np.full(n, np.nan)
        sn = np.full(n, np.nan)
        p = m = 0.0
        for i in range(n):
            if np.isnan(x[i]):
                continue
            z = (x[i] - mu) / sigma
            p = max(0.0, p + z - self.k)
            m = max(0.0, m - z - self.k)
            sp[i], sn[i] = p, m
            if self.reset and (p > self.h or m > self.h):
                pass  # keep accumulating; the finding interval spans the alarm run
        self.pos_, self.neg_ = sp, sn
        score = np.maximum(sp, sn) / self.h
        unknown = np.isnan(score)
        return TruthSeries.from_bool(np.where(unknown, False, score > 1.0), unknown), score

    def describe(self, dataset, start, end, evidence):
        return ("Sustained shift (CUSUM):\nThe accumulated deviation from the\nreference level exceeds the alarm limit.\n\n"
                "Question answered:\nHas the level shifted persistently? Yes.")

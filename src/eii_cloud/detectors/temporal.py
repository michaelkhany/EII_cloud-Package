"""Temporal detectors: flatline, missingness, rate of change, change point."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..findings import Category
from ..truth import TruthSeries
from .base import BaseDetector, register_detector


@register_detector("flatline")
class FlatlineDetector(BaseDetector):
    """A3 abnormal flatline: the value stays within ``tolerance`` of a run reference for at
    least ``min_run`` consecutive observed samples.

    ``tolerance`` should be at least the measurement quantisation step.  With ``regimes``
    the detector only fires inside the listed context regimes (e.g. where variation is
    physically expected), and with ``min_expected_width`` it abstains (FALSE) where the
    envelope named by ``envelope`` is itself narrower than that width.
    """

    category = Category.FLATLINE
    default_visualization = "EIICloudChart"

    def __init__(self, parameter: str, min_run: int = 8, tolerance: float = 0.0, regimes: list[str] | None = None,
                 envelope: str | None = None, min_expected_width: float | None = None, **kw: Any):
        super().__init__(parameter, min_run=min_run, tolerance=tolerance, regimes=regimes, envelope=envelope,
                         min_expected_width=min_expected_width, **kw)
        self.min_run, self.tolerance, self.regimes = min_run, tolerance, regimes
        self.envelope_name, self.min_expected_width = envelope, min_expected_width

    def evaluate(self, dataset, envelopes, context):
        x = dataset.value(self.parameter)
        n = len(x)
        obs = ~np.isnan(x)
        flat = np.zeros(n, dtype=bool)
        run_len = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            if not obs[i]:
                i += 1
                continue
            j = i
            ref = x[i]
            while j + 1 < n and obs[j + 1] and abs(x[j + 1] - ref) <= self.tolerance:
                j += 1
            length = j - i + 1
            if length >= self.min_run:
                flat[i:j + 1] = True
                run_len[i:j + 1] = length
            i = j + 1
        unknown = ~obs
        if self.regimes and context is not None:
            labels = context.regime_labels(dataset)
            in_regime = np.isin(labels.astype(str), self.regimes)
            unknown |= labels.astype(str) == "__unknown__"
            flat &= in_regime
        if self.envelope_name is not None and self.min_expected_width is not None:
            env = envelopes[self.envelope_name]
            narrow = env.support & (env.width < self.min_expected_width)
            flat &= ~narrow
        return TruthSeries.from_bool(flat, unknown), run_len.astype(float)

    def evidence_for(self, dataset, start, end, score):
        ev = super().evidence_for(dataset, start, end, score)
        ev.update({"run_length": int(end - start + 1), "tolerance": self.tolerance,
                   "constant_value": float(dataset.value(self.parameter)[start])})
        return ev

    def describe(self, dataset, start, end, evidence):
        return ("Abnormal flatline:\nThe value remains almost constant.\nThis may indicate a frozen sensor,\n"
                "stalled process, or exporter issue.\n\nQuestions answered:\nIs it changing normally? No.\n"
                "Is it too flat when it should vary? Yes.")


@register_detector("missingness")
class MissingnessDetector(BaseDetector):
    """A2 missing/unobservable: the parameter is absent while the acquisition schedule
    expects it.  Scheduled gaps (``dataset.expected == False``) are FALSE, not findings.

    With ``max_gap`` the detector also reports gaps in the time axis itself (no sample at
    all) longer than ``max_gap``, anchored on the sample after the gap.
    """

    category = Category.MISSING
    default_visualization = "MissingnessTimeline"

    def __init__(self, parameter: str, min_gap: int = 1, max_gap: float | str | None = None, **kw: Any):
        super().__init__(parameter, min_gap=min_gap, max_gap=max_gap, min_duration=min_gap, **kw)
        self.max_gap = max_gap

    def evaluate(self, dataset, envelopes, context):
        missing = dataset.missing_when_expected(self.parameter)
        score = missing.astype(float)
        if self.max_gap is not None and dataset.n > 1:
            gap = np.diff(dataset.time.t)
            long = np.concatenate(([False], gap > dataset.time.offset(self.max_gap)))
            score = score + 2 * long
            missing = missing | long
        return TruthSeries.from_bool(missing), score

    def evidence_for(self, dataset, start, end, score):
        ev = {"parameter": self.parameter, "n_samples": int(end - start + 1),
              "expected": True, "observed": False,
              "last_observed_before": None, "first_observed_after": None}
        x = dataset.value(self.parameter)
        before = np.flatnonzero(~np.isnan(x[:start]))
        after = np.flatnonzero(~np.isnan(x[end + 1:]))
        if len(before):
            ev["last_observed_before"] = dataset.time.label(int(before[-1]))
        if len(after):
            ev["first_observed_after"] = dataset.time.label(int(end + 1 + after[0]))
        if score is not None and np.any(score[start:end + 1] >= 2):
            ev["time_axis_gap"] = True
        return ev

    def anchor(self, dataset, start, end, score):
        return (start + end) // 2, None

    def describe(self, dataset, start, end, evidence):
        return ("Observability degradation:\nThe parameter disappears\nalthough it should exist.\n\n"
                "Questions answered:\nIs the value present? No.\nIs it missing when expected? Yes.")


@register_detector("rate_of_change")
class RateOfChangeDetector(BaseDetector):
    """A5 temporal instability: absolute first derivative above ``max_rate`` (units per time unit)."""

    category = Category.TEMPORAL_INSTABILITY
    default_visualization = "AnomalyScoreChart"

    def __init__(self, parameter: str, max_rate: float, per: float | str = 1, **kw: Any):
        super().__init__(parameter, max_rate=max_rate, per=per, **kw)
        self.max_rate, self.per = max_rate, per

    def evaluate(self, dataset, envelopes, context):
        x = dataset.value(self.parameter)
        t = dataset.time.t
        rate = np.full(len(x), np.nan)
        if len(x) > 1:
            dt = np.diff(t) / dataset.time.offset(self.per)
            rate[1:] = np.diff(x) / dt
        score = np.abs(rate) / self.max_rate
        unknown = np.isnan(score)
        return TruthSeries.from_bool(np.where(unknown, False, score > 1.0), unknown), score

    def describe(self, dataset, start, end, evidence):
        return (f"Temporal instability:\nThe value changes faster than\n{self.max_rate} per {self.per} time unit(s).\n\n"
                "Question answered:\nIs the transition plausible? No.")


@register_detector("change_point")
class ChangePointDetector(BaseDetector):
    """Offline mean change-point detection by binary segmentation with a BIC-type penalty.

    Non-causal (uses the full series); not available for streaming consistency.  Each change
    point produces a single-sample finding anchored at the first sample of the new segment.
    """

    category = Category.TEMPORAL_INSTABILITY
    default_visualization = "ChangePointChart"
    causal = False

    def __init__(self, parameter: str, penalty: float | None = None, min_size: int = 5, max_changes: int = 20,
                 **kw: Any):
        super().__init__(parameter, penalty=penalty, min_size=min_size, max_changes=max_changes, **kw)
        self.penalty, self.min_size, self.max_changes = penalty, min_size, max_changes
        self.segments_: list[tuple[int, int, float]] = []

    def evaluate(self, dataset, envelopes, context):
        x = dataset.value(self.parameter)
        obs = np.flatnonzero(~np.isnan(x))
        y = x[obs]
        n = len(y)
        cps: list[int] = []
        if n >= 2 * self.min_size:
            sigma2 = max(float(np.var(np.diff(y)) / 2.0), 1e-12)  # robust-ish noise variance
            pen = self.penalty if self.penalty is not None else 3.0 * np.log(n) * sigma2
            cps = _binary_segmentation(y, self.min_size, pen, self.max_changes)
        truth = np.zeros(len(x), dtype=bool)
        score = np.full(len(x), np.nan)
        bounds = [0] + sorted(cps) + [n]
        self.segments_ = [(int(obs[bounds[i]]), int(obs[bounds[i + 1] - 1]), float(np.mean(y[bounds[i]:bounds[i + 1]])))
                          for i in range(len(bounds) - 1)]
        for cp in cps:
            truth[obs[cp]] = True
            score[obs[cp]] = abs(np.mean(y[cp:]) - np.mean(y[:cp]))
        return TruthSeries.from_bool(truth, np.isnan(x)), score

    def evidence_for(self, dataset, start, end, score):
        ev = super().evidence_for(dataset, start, end, score)
        for s0, _s1, mean in self.segments_:
            if s0 == start:
                prev = [m for a, b, m in self.segments_ if b == s0 - 1 or b < s0]
                ev.update({"segment_mean_after": mean, "segment_mean_before": prev[-1] if prev else None})
        return ev

    def describe(self, dataset, start, end, evidence):
        return ("Regime transition:\nThe mean level changes at this point\n(offline change-point detection).\n\n"
                "Question answered:\nIs the behaviour stationary here? No.")


def _binary_segmentation(y: np.ndarray, min_size: int, penalty: float, max_changes: int) -> list[int]:
    """Greedy binary segmentation on the L2 cost with a fixed penalty per change."""
    cps: list[int] = []
    segments = [(0, len(y))]
    while segments and len(cps) < max_changes:
        best = None
        for (a, b) in segments:
            if b - a < 2 * min_size:
                continue
            seg = y[a:b]
            m = len(seg)
            cs = np.cumsum(seg)
            cs2 = np.cumsum(seg ** 2)
            total = cs2[-1] - cs[-1] ** 2 / m
            ks = np.arange(min_size, m - min_size + 1)
            left = cs2[ks - 1] - cs[ks - 1] ** 2 / ks
            right = (cs2[-1] - cs2[ks - 1]) - (cs[-1] - cs[ks - 1]) ** 2 / (m - ks)
            gain = total - (left + right)
            j = int(np.argmax(gain))
            if gain[j] > penalty and (best is None or gain[j] > best[0]):
                best = (gain[j], a + int(ks[j]), a, b)
        if best is None:
            break
        _, cp, a, b = best
        cps.append(cp)
        segments.remove((a, b))
        segments.extend([(a, cp), (cp, b)])
    return sorted(cps)

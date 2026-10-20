"""Vectorised three-valued window aggregations on an event-time axis."""
from __future__ import annotations

import numpy as np

from ..truth import TruthSeries


def _bounds(t: np.ndarray, back: float, fwd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.searchsorted(t, t - back, side="left")
    hi = np.searchsorted(t, t + fwd, side="right")
    incomplete = (t - back < t[0]) | (t + fwd > t[-1])
    return lo, hi, incomplete


def _cum(mask: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(mask.astype(np.int64))))


def window_any(ts: TruthSeries, t: np.ndarray, back: float, fwd: float) -> TruthSeries:
    """TRUE at i if any TRUE sample lies in ``[t_i - back, t_i + fwd]``.

    FALSE only when the window is fully inside the observed axis and contains neither
    TRUE nor UNKNOWN samples; otherwise UNKNOWN (insufficient evidence).
    """
    v = ts.values
    lo, hi, incomplete = _bounds(t, back, fwd)
    cT, cU = _cum(v == 1), _cum(v == -1)
    any_t = (cT[hi] - cT[lo]) > 0
    any_u = (cU[hi] - cU[lo]) > 0
    out = np.where(any_t, 1, np.where(any_u | incomplete, -1, 0))
    return TruthSeries(out)


def window_all(ts: TruthSeries, t: np.ndarray, back: float) -> TruthSeries:
    """TRUE at i if every sample in ``[t_i - back, t_i]`` is TRUE and the window is fully
    covered by the axis; FALSE if any sample is FALSE; UNKNOWN otherwise."""
    v = ts.values
    lo, hi, incomplete = _bounds(t, back, 0.0)
    cF, cU = _cum(v == 0), _cum(v == -1)
    any_f = (cF[hi] - cF[lo]) > 0
    any_u = (cU[hi] - cU[lo]) > 0
    out = np.where(any_f, 0, np.where(any_u | incomplete, -1, 1))
    return TruthSeries(out)


def onsets(ts: TruthSeries) -> np.ndarray:
    v = ts.values
    prev = np.concatenate(([0], v[:-1]))
    return (v == 1) & (prev != 1)


def window_count(ts: TruthSeries, t: np.ndarray, back: float, min_count: int) -> tuple[TruthSeries, np.ndarray]:
    """Count TRUE *onsets* in ``[t_i - back, t_i]``; TRUE when at least ``min_count``."""
    v = ts.values
    lo, hi, incomplete = _bounds(t, back, 0.0)
    cO, cU = _cum(onsets(ts)), _cum(v == -1)
    count = cO[hi] - cO[lo]
    any_u = (cU[hi] - cU[lo]) > 0
    out = np.where(count >= min_count, 1, np.where(any_u | incomplete, -1, 0))
    return TruthSeries(out), count


def run_level(a: TruthSeries, b: TruthSeries, mode: str) -> TruthSeries:
    """Interval semantics on maximal TRUE runs of ``a``.

    ``mode='during'``: the whole run must be inside TRUE samples of ``b``.
    ``mode='overlaps'``: at least one sample of the run is TRUE in ``b``.
    Outside runs the result follows ``a`` (FALSE / UNKNOWN).
    """
    va, vb = a.values, b.values
    out = np.where(va == 1, -1, va).astype(np.int8)
    for s, e in a.runs():
        seg = vb[s:e + 1]
        if mode == "during":
            val = 0 if np.any(seg == 0) else (1 if np.all(seg == 1) else -1)
        elif mode == "overlaps":
            val = 1 if np.any(seg == 1) else (-1 if np.any(seg == -1) else 0)
        else:
            raise ValueError(mode)
        out[s:e + 1] = val
    return TruthSeries(out)

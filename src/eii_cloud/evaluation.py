"""Evaluation metrics for event-level, interval-level and classification experiments.

All functions are pure NumPy; uncertainty intervals use bootstrap over *independent units*
(assets, incidents) supplied by the caller rather than over correlated samples.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .findings import Finding

Interval = tuple[int, int]


@dataclass
class IntervalMatch:
    truth: Interval
    matched: list[str] = field(default_factory=list)
    exact: bool = False
    first_detection_index: int | None = None
    detection_delay: int | None = None  # samples after truth start (negative = early warning)


def match_intervals(findings: Iterable[Finding], truth: Sequence[Interval], tolerance_before: int = 0,
                    tolerance_after: int = 0) -> tuple[list[IntervalMatch], list[Finding]]:
    """Match findings to ground-truth sample intervals.

    A finding matches a truth interval when their sample ranges overlap after widening
    the truth interval by ``tolerance_before``/``tolerance_after``. Returns the matches and
    the list of unmatched (false-alert) findings.
    """
    finds = list(findings)
    matched_ids: set[str] = set()
    matches: list[IntervalMatch] = []
    for (s, e) in truth:
        m = IntervalMatch((s, e))
        lo, hi = s - tolerance_before, e + tolerance_after
        for f in finds:
            if f.overlaps(lo, hi):
                m.matched.append(f.finding_id)
                matched_ids.add(f.finding_id)
                if f.start_index == s and f.end_index == e:
                    m.exact = True
                if m.first_detection_index is None or f.start_index < m.first_detection_index:
                    m.first_detection_index = f.start_index
        if m.first_detection_index is not None:
            m.detection_delay = m.first_detection_index - s
        matches.append(m)
    unmatched = [f for f in finds if f.finding_id not in matched_ids]
    return matches, unmatched


def event_metrics(findings: Iterable[Finding], truth: Sequence[Interval], n_samples: int,
                  tolerance_before: int = 0, tolerance_after: int = 0,
                  truth_series: Sequence[int] | None = None) -> dict[str, Any]:
    """Event recall/precision, exact-interval accuracy, delays, alert burden, abstention."""
    finds = list(findings)
    matches, unmatched = match_intervals(finds, truth, tolerance_before, tolerance_after)
    n_truth = len(truth)
    detected = [m for m in matches if m.matched]
    delays = [m.detection_delay for m in detected if m.detection_delay is not None]
    exact = sum(1 for m in matches if m.exact)
    out: dict[str, Any] = {
        "n_truth": n_truth,
        "n_findings": len(finds),
        "n_detected": len(detected),
        "event_recall": len(detected) / n_truth if n_truth else float("nan"),
        "event_precision": (len(finds) - len(unmatched)) / len(finds) if finds else float("nan"),
        "exact_interval_accuracy": exact / n_truth if n_truth else float("nan"),
        "false_alert_episodes": len(unmatched),
        "false_alerts_per_1000_samples": 1000.0 * len(unmatched) / n_samples if n_samples else float("nan"),
        "alert_burden_per_1000_samples": 1000.0 * len(finds) / n_samples if n_samples else float("nan"),
        "mean_detection_delay": float(np.mean(delays)) if delays else float("nan"),
        "max_detection_delay": int(np.max(delays)) if delays else None,
        "first_warning_lead_time": float(-np.min(delays)) if delays else float("nan"),
        "matches": [{"truth": m.truth, "matched": m.matched, "exact": m.exact, "delay": m.detection_delay} for m in matches],
        "unmatched_finding_ids": [f.finding_id for f in unmatched],
    }
    if truth_series is not None:
        ts = np.asarray(truth_series, dtype=np.int8)
        out["abstention_rate"] = float(np.mean(ts == -1))
        out["rule_coverage"] = float(np.mean(ts != -1))
    return out


def confusion_matrix(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any] | None = None) -> tuple[np.ndarray, list]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    labs = list(labels) if labels is not None else sorted(set(yt.tolist()) | set(yp.tolist()))
    idx = {l: i for i, l in enumerate(labs)}
    cm = np.zeros((len(labs), len(labs)), dtype=int)
    for a, b in zip(yt.tolist(), yp.tolist()):
        cm[idx[a], idx[b]] += 1
    return cm, labs


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any] | None = None) -> dict[str, Any]:
    """Macro-F1, balanced accuracy, per-class precision/recall and the confusion matrix."""
    cm, labs = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = (tp + fn) > 0
    return {
        "labels": labs,
        "confusion_matrix": cm.tolist(),
        "macro_f1": float(f1[present].mean()) if present.any() else float("nan"),
        "balanced_accuracy": float(rec[present].mean()) if present.any() else float("nan"),
        "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else float("nan"),
        "per_class": {str(l): {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for l, p, r, f, s in zip(labs, prec, rec, f1, tp + fn)},
    }


def bootstrap_ci(units: Sequence[Any], statistic: Callable[[Sequence[Any]], float], n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Percentile bootstrap CI of ``statistic`` resampling *independent units* with replacement."""
    units = list(units)
    rng = np.random.default_rng(seed)
    if not units:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan")}
    est = float(statistic(units))
    boots = np.array([statistic([units[i] for i in rng.integers(0, len(units), len(units))]) for _ in range(n_boot)])
    return {"estimate": est, "low": float(np.quantile(boots, alpha / 2)), "high": float(np.quantile(boots, 1 - alpha / 2)),
            "n_units": len(units), "n_boot": n_boot}


def per_category_detection(findings: Iterable[Finding], truth: Mapping[str, Sequence[Interval]], n_samples: int,
                           **kw: Any) -> dict[str, dict[str, Any]]:
    """Event metrics per anomaly category (keys of ``truth`` are category values)."""
    finds = list(findings)
    out = {}
    for cat, intervals in truth.items():
        subset = [f for f in finds if f.category.value == cat]
        out[cat] = event_metrics(subset, intervals, n_samples, **kw)
    return out


def monitoring_vs_physical(findings: Iterable[Finding], injected_monitoring: Sequence[Interval],
                           injected_physical: Sequence[Interval]) -> dict[str, Any]:
    """EXP-03: how often monitoring faults (missing/flatline) are mislabelled as physical anomalies."""
    monitoring_cats = {"missing_when_expected", "abnormal_flatline"}
    finds = list(findings)
    confused = 0
    for (s, e) in injected_monitoring:
        if any(f.overlaps(s, e) and f.category.value not in monitoring_cats and f.rule_id is None for f in finds):
            confused += 1
    recovered = sum(1 for (s, e) in injected_monitoring
                    if any(f.overlaps(s, e) and f.category.value in monitoring_cats for f in finds))
    phys = sum(1 for (s, e) in injected_physical
               if any(f.overlaps(s, e) and f.category.value not in monitoring_cats for f in finds))
    return {
        "n_monitoring_faults": len(injected_monitoring),
        "monitoring_faults_recovered": recovered,
        "monitoring_faults_flagged_as_physical": confused,
        "n_physical_faults": len(injected_physical),
        "physical_faults_detected": phys,
    }

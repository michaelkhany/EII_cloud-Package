"""Controlled synthetic scenarios.

:func:`make_reference_scenario` reproduces the *behaviours* shown in the reference EII
Cloud figure (normal operation, sustained deviation, missing telemetry, abnormal
flatline, context contradiction) with exactly known intervals.  The numbers are generated,
not read off the image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..context import ContextDefinition
from ..data import EIIDataset
from ..evidence.events import EventLog
from ..rules import RuleSet

HALF_WIDTH = 3.5
UTIL_GAIN = 0.15  # degrees per utilisation point above the moderate level
UTIL_MODERATE = 55.0


def expected_center(t: np.ndarray) -> np.ndarray:
    return 62.0 + 3.0 * np.sin(2 * np.pi * np.asarray(t, dtype=float) / 50.0)


@dataclass
class Scenario:
    dataset: EIIDataset
    reference: EIIDataset
    context: ContextDefinition
    ground_truth: dict[str, list[tuple[int, int]]]
    envelopes: dict[str, dict[str, Any]]
    rules: RuleSet
    description: str = ""
    injected: dict[str, Any] = field(default_factory=dict)


def _context_definition() -> ContextDefinition:
    cd = ContextDefinition()
    cd.add_regime("high_load", "gpu_util", ">=", 85)
    cd.expect(regime="high_load", parameter="gpu_temp", kind="min_level", value=63.0, settle=3,
              description="Under sustained high workload GPU temperature must rise above 63 degC after a 3-sample settle time")
    return cd


def _rules() -> RuleSet:
    return RuleSet.from_dict({
        "schema_version": "1.0",
        "name": "reference_rules",
        "rules": [
            {
                "id": "deviation_then_gpu_error",
                "category": "composite_evidence",
                "severity": "high",
                "parameters": ["gpu_temp"],
                "description": ("Composite evidence:\nTemperature leaves its validity envelope\nand a GPU error is logged "
                                "within 12 windows.\n\nQuestion answered:\nDo deviation and error co-occur? Yes."),
                "expression": {"before": {
                    "condition": {"outside_envelope": {"parameter": "gpu_temp", "envelope": "gpu_temp_static"}},
                    "next": {"event": "gpu_xid"},
                    "window": 12}},
            },
            {
                "id": "instability_under_load",
                "category": "composite_evidence",
                "severity": "critical",
                "parameters": ["gpu_temp", "gpu_util"],
                "description": ("Composite evidence:\nSustained high workload while the\nworkload-conditioned envelope is violated.\n\n"
                                "Question answered:\nIs the GPU unstable under load? Yes."),
                "expression": {"and": [
                    {"persists_for": {"condition": {"signal": "gpu_util", "op": ">=", "value": 85}, "window": 3}},
                    {"outside_envelope": {"parameter": "gpu_temp", "envelope": "gpu_temp_ctx"}},
                ]},
            },
            {
                "id": "error_without_telemetry",
                "category": "missing_when_expected",
                "severity": "medium",
                "parameters": ["gpu_temp"],
                "description": "Monitoring problem:\nGPU error logs cannot be assessed\nwhile telemetry is missing.",
                "expression": {"and": [
                    {"missing_when_expected": "gpu_temp"},
                    {"not": {"event": "gpu_xid"}},
                ]},
            },
        ],
    })


def make_reference_scenario(seed: int = 0, n: int = 100, noise: float = 0.9) -> Scenario:
    """Synthetic single-GPU scenario with the five behaviours of the reference chart.

    Ground-truth sample intervals (inclusive):

    * ``value_deviation``       33..44  (ramp above the static envelope, GPU error at 44)
    * ``missing_when_expected`` 50..56  (telemetry absent although scheduled; log collector down)
    * ``abnormal_flatline``     66..75  (constant value *inside* the envelope)
    * ``context_contradiction`` 83..92  (high load 80..92, temperature fails to rise; 3-sample settle)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    center = expected_center(t)
    lower, upper = center - HALF_WIDTH, center + HALF_WIDTH
    util = UTIL_MODERATE + rng.normal(0, 4, n)
    temp = center + rng.normal(0, noise, n)

    dev = (t >= 33) & (t <= 44)
    temp[dev] = upper[dev] + 1.5 + 0.5 * (t[dev] - 33) + rng.normal(0, 0.3, dev.sum())
    miss = (t >= 50) & (t <= 56)
    temp[miss] = np.nan
    flat = (t >= 66) & (t <= 75)
    temp[flat] = 61.5
    load = (t >= 80) & (t <= 92)
    util[load] = 95 + rng.normal(0, 1.5, load.sum())
    # temperature does NOT respond to the load: stays at its unloaded baseline (already set)

    expected_temp = center  # documented physical expectation (e.g. from a thermal model / schedule)
    ds = EIIDataset(
        time=t, values={"gpu_temp": temp, "gpu_util": util},
        context={"expected_temp": expected_temp, "gpu_util": util},
        units={"gpu_temp": "degC", "gpu_util": "%"},
        events=EventLog.from_records(
            [{"event_id": "xid-001", "event_type": "gpu_xid", "timestamp": 44, "asset_id": "gpu0", "source": "gpu_xid",
              "attrs": "Xid 79: GPU has fallen off the bus"}],
            coverage=[{"source": "gpu_xid", "start": 0, "end": 49}, {"source": "gpu_xid", "start": 57, "end": n - 1}],
        ),
        asset_id="gpu0", name=f"reference_scenario_seed{seed}",
        attrs={"generator": "eii_cloud.experiments.synthetic.make_reference_scenario", "seed": seed, "n": n, "noise": noise},
    )
    reference = make_reference_period(seed=seed + 1000, n=600, noise=noise)
    envelopes = {
        "gpu_temp_static": {"method": "external_envelope", "parameter": "gpu_temp", "lower": lower.tolist(),
                            "upper": upper.tolist(), "center": center.tolist(), "times": t.tolist(),
                            "source": "documented expected trajectory"},
        "gpu_temp_ctx": {"method": "physical_constraints", "parameter": "gpu_temp",
                         "lower": {"expected_temp": 1.0, "gpu_util": UTIL_GAIN, "const": -HALF_WIDTH - UTIL_GAIN * UTIL_MODERATE},
                         "upper": {"expected_temp": 1.0, "gpu_util": UTIL_GAIN, "const": HALF_WIDTH - UTIL_GAIN * UTIL_MODERATE},
                         "center": {"expected_temp": 1.0, "gpu_util": UTIL_GAIN, "const": -UTIL_GAIN * UTIL_MODERATE}},
    }
    truth = {
        "value_deviation": [(33, 44)],
        "missing_when_expected": [(50, 56)],
        "abnormal_flatline": [(66, 75)],
        "context_contradiction": [(83, 92)],
    }
    return Scenario(ds, reference, _context_definition(), truth, envelopes, _rules(),
                    description="Reference EII Cloud scenario (EXP-00)",
                    injected={"high_load": (80, 92), "gpu_error_at": 44, "log_coverage_gap": (50, 56)})


def make_reference_period(seed: int = 1, n: int = 600, noise: float = 0.9) -> EIIDataset:
    """Clean reference data where temperature *does* respond to workload (for fitting envelopes)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    center = expected_center(t)
    util = np.clip(UTIL_MODERATE + rng.normal(0, 4, n), 0, 100)
    for s in range(60, n, 150):  # periodic load phases
        util[s:s + 25] = 95 + rng.normal(0, 1.5, min(25, n - s))
    temp = center + UTIL_GAIN * (util - UTIL_MODERATE) + rng.normal(0, noise, n)
    return EIIDataset(time=t, values={"gpu_temp": temp, "gpu_util": util},
                      context={"expected_temp": center, "gpu_util": util},
                      units={"gpu_temp": "degC", "gpu_util": "%"}, asset_id="gpu0",
                      name=f"reference_period_seed{seed}")


def inject(dataset: EIIDataset, parameter: str, kind: str, start: int, end: int, value: float | None = None,
           seed: int = 0) -> EIIDataset:
    """Return a copy with a controlled disturbance on ``parameter`` over samples ``start..end``.

    ``kind``: ``missing`` | ``flatline`` | ``offset`` (add ``value``) | ``spike`` (add ``value``
    at ``start`` only) | ``delay`` (shift the segment ``value`` samples later) |
    ``not_expected`` (mark the window as a scheduled gap).
    """
    vals = dataset.values.copy()
    exp = dataset.expected.copy()
    x = vals[parameter].to_numpy(dtype=float).copy()
    sl = slice(start, end + 1)
    if kind == "missing":
        x[sl] = np.nan
    elif kind == "flatline":
        x[sl] = x[start] if value is None else value
    elif kind == "offset":
        x[sl] = x[sl] + (value if value is not None else 0.0)
    elif kind == "spike":
        x[start] = x[start] + (value if value is not None else 0.0)
    elif kind == "delay":
        k = int(value or 1)
        seg = x[sl].copy()
        x[sl] = np.nan
        x[start + k:end + 1 + k] = seg[:max(0, len(seg) - max(0, start + k + len(seg) - len(x)))]
    elif kind == "not_expected":
        x[sl] = np.nan
        exp.loc[start:end, parameter] = False
    else:
        raise ValueError(f"unknown injection kind {kind!r}")
    vals[parameter] = x
    return EIIDataset(dataset.time.index, vals, dataset.context if len(dataset.context.columns) else None, exp,
                      units=dataset.units, events=dataset.events, asset_id=dataset.asset_id,
                      name=f"{dataset.name}+{kind}[{start}-{end}]",
                      attrs={**dataset.attrs, "injection": {"parameter": parameter, "kind": kind, "start": start,
                                                             "end": end, "value": value}})

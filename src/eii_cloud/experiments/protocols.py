"""Built-in scientific validation protocols (EXP-00 ... EXP-06) on the synthetic scenario.

Each function returns a plain dict report.  They are deliberately small, deterministic and
fast so they double as end-to-end tests; real-data studies should build on
:class:`~eii_cloud.experiments.runner.ExperimentRunner` instead.
"""
from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis import EIIEngine, StreamingEngine, split_chunks
from ..evaluation import event_metrics, monitoring_vs_physical, per_category_detection
from ..export import ExportManager, svg_ids
from ..findings import findings_equal
from ..visualization.annotations import annotations_document
from ..visualization.eii_chart import EIICloudChart
from .runner import default_detectors
from .synthetic import inject, make_reference_scenario


def _engine(sc, envelopes=None, detectors=None, rules=None) -> EIIEngine:
    envs = envelopes if envelopes is not None else sc.envelopes
    dets = detectors if detectors is not None else default_detectors(list(envs))
    return EIIEngine(envelopes=envs, detectors=dets, rules=rules if rules is not None else sc.rules, context=sc.context)


def exp00_reference_reproduction(seed: int = 0) -> dict[str, Any]:
    """EXP-00: exact interval/category reproduction of the reference behaviours."""
    sc = make_reference_scenario(seed)
    result = _engine(sc).run(sc.dataset, sc.reference)
    det_findings = [f for f in result.findings if f.rule_id is None]
    per_cat = per_category_detection(det_findings, sc.ground_truth, sc.dataset.n)
    exact = all(v["exact_interval_accuracy"] == 1.0 and v["false_alert_episodes"] == 0 for v in per_cat.values())
    anns = annotations_document(result, sc.dataset, "gpu_temp")
    return {"experiment": "EXP-00", "seed": seed, "all_intervals_exact": exact,
            "per_category": {k: {kk: v[kk] for kk in ("event_recall", "exact_interval_accuracy", "false_alert_episodes")}
                             for k, v in per_cat.items()},
            "n_rule_findings": sum(1 for f in result.findings if f.rule_id is not None),
            "rule_findings": [(f.rule_id, f.start_index, f.end_index) for f in result.findings if f.rule_id],
            "n_annotations": anns["n_annotations"], "analysis_id": result.analysis_id}


def exp01_context_vs_static(seed: int = 0) -> dict[str, Any]:
    """EXP-01: static threshold vs time-varying vs context-conditioned envelope (+ composite rules)."""
    sc = make_reference_scenario(seed)
    truth_all = [iv for k, ivs in sc.ground_truth.items() if k != "missing_when_expected" and k != "abnormal_flatline"
                 for iv in ivs]
    variants = {
        "static_threshold": {"gpu_temp_fixed": {"method": "fixed_bounds", "parameter": "gpu_temp", "lower": 55.0, "upper": 69.0}},
        "time_varying_envelope": {"gpu_temp_static": sc.envelopes["gpu_temp_static"]},
        "context_conditioned": {"gpu_temp_ctx": sc.envelopes["gpu_temp_ctx"]},
    }
    report: dict[str, Any] = {"experiment": "EXP-01", "seed": seed, "variants": {}}
    for name, envs in variants.items():
        env_name = next(iter(envs))
        dets = [{"type": "threshold", "parameter": "gpu_temp", "envelope": env_name, "detector_id": "thr"}]
        res = _engine(sc, envs, dets, rules=sc.rules.__class__()).run(sc.dataset, sc.reference)
        m = event_metrics([f for f in res.findings if f.detector_id == "thr"], truth_all, sc.dataset.n,
                          truth_series=res.truth_series["thr"])
        cov = res.envelopes[env_name]["coverage"]
        report["variants"][name] = {"event_recall": m["event_recall"], "event_precision": m["event_precision"],
                                    "false_alert_episodes": m["false_alert_episodes"],
                                    "abstention_rate": m["abstention_rate"], "coverage": cov}
    full = _engine(sc).run(sc.dataset, sc.reference)
    m = event_metrics([f for f in full.findings if f.rule_id is not None], truth_all, sc.dataset.n)
    report["variants"]["context_plus_composite_rules"] = {"event_recall": m["event_recall"],
                                                          "event_precision": m["event_precision"],
                                                          "false_alert_episodes": m["false_alert_episodes"]}
    return report


def exp03_observability_degradation(seed: int = 0) -> dict[str, Any]:
    """EXP-03: injected monitoring faults must be reported as such, not as physical anomalies."""
    sc = make_reference_scenario(seed)
    base = sc.dataset
    from .synthetic import expected_center
    # frozen value chosen inside the envelope so the fault is purely a monitoring fault
    cases = [("missing", 10, 16, None), ("flatline", 20, 30, float(expected_center(np.array([25]))[0])),
             ("not_expected", 58, 62, None)]
    ds = base
    for kind, s, e, v in cases:
        ds = inject(ds, "gpu_temp", kind, s, e, v)
    result = _engine(sc).run(ds, sc.reference)
    det = [f for f in result.findings if f.rule_id is None]
    rep = monitoring_vs_physical(det, [(10, 16), (20, 30)], [(33, 44)])
    scheduled_gap_flagged = any(f.overlaps(58, 62) and f.category.value == "missing_when_expected" for f in det)
    return {"experiment": "EXP-03", "seed": seed, **rep, "scheduled_gap_flagged_as_missing": scheduled_gap_flagged,
            "findings": [(f.category.value, f.start_index, f.end_index) for f in det]}


def exp05_export_reproducibility(seed: int = 0, out_dir: str | Path | None = None) -> dict[str, Any]:
    """EXP-05: identical findings/ids across modes and formats; JSON independent of rendering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = make_reference_scenario(seed)
    engine = _engine(sc)
    result = engine.run(sc.dataset, sc.reference)
    json_before = annotations_document(result, sc.dataset, "gpu_temp")
    tmp = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="eii_exp05_"))
    em = ExportManager(tmp, dpi=72)
    ids_per_mode: dict[str, list[str]] = {}
    svg_ok = True
    for mode in ("annotated", "clean", "flags_only"):
        ch = EIICloudChart(sc.dataset, result, "gpu_temp", mode=mode)
        fig = ch.render()
        files = em.export(fig, f"gpu_temp_{mode}", formats=("svg", "pdf", "tiff", "png"))
        plt.close(fig)
        ids_per_mode[mode] = ch.annotation_ids()
        if mode != "clean":
            present = svg_ids(files["svg"])
            expected = {f"flag-{a}" for a in ch.annotation_ids()} | ({a for a in ch.annotation_ids()} if mode == "annotated" else set())
            svg_ok &= expected.issubset(present)
    json_after = annotations_document(result, sc.dataset, "gpu_temp")
    result2 = engine.run(sc.dataset, sc.reference)
    return {"experiment": "EXP-05", "seed": seed, "out_dir": str(tmp),
            "annotation_ids_identical_across_modes": len({tuple(v) for v in ids_per_mode.values()}) == 1,
            "svg_contains_annotation_ids": svg_ok,
            "json_unchanged_by_rendering": json.dumps(json_before, sort_keys=True, default=str) == json.dumps(json_after, sort_keys=True, default=str),
            "findings_unchanged_by_rerun": findings_equal(result.findings, result2.findings),
            "missing_regions": json_after["missing_regions"]}


def exp06_batch_vs_streaming(seed: int = 0, chunk_size: int = 10) -> dict[str, Any]:
    """EXP-06: finalised streaming findings equal batch findings (causal engine)."""
    sc = make_reference_scenario(seed)
    engine = _engine(sc)
    batch = engine.run(sc.dataset, sc.reference)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stream = StreamingEngine(engine, sc.reference)
        n_prov_max = 0
        premature_changes = 0
        finals_seen: dict[str, Any] = {}
        for chunk in split_chunks(sc.dataset, chunk_size):
            res = stream.push(chunk)
            n_prov_max = max(n_prov_max, sum(1 for f in res.findings if f.provisional))
            for f in res.findings:
                if not f.provisional:
                    if f.finding_id in finals_seen and finals_seen[f.finding_id] != (f.start_index, f.end_index):
                        premature_changes += 1
                    finals_seen[f.finding_id] = (f.start_index, f.end_index)
        final = stream.flush()
    batch_ids = sorted(f.finding_id for f in batch.findings)
    final_ids = sorted(f.finding_id for f in final.findings if not f.provisional)
    pre_flush_ids = sorted(finals_seen)
    return {"experiment": "EXP-06", "seed": seed, "chunk_size": chunk_size,
            "identical_after_flush": batch_ids == final_ids,
            "finalised_before_flush_subset_of_batch": set(pre_flush_ids).issubset(batch_ids),
            "finalised_findings_never_changed": premature_changes == 0,
            "n_batch": len(batch_ids), "n_final_before_flush": len(pre_flush_ids), "max_provisional": n_prov_max}


def run_all(seed: int = 0) -> dict[str, Any]:
    return {"EXP-00": exp00_reference_reproduction(seed), "EXP-01": exp01_context_vs_static(seed),
            "EXP-03": exp03_observability_degradation(seed), "EXP-05": exp05_export_reproducibility(seed),
            "EXP-06": exp06_batch_vs_streaming(seed)}

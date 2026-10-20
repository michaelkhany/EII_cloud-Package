"""Reproduce the reference EII Cloud figure end to end (EXP-00) and export every artefact.

Run:  python examples/reference_chart.py out_dir
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eii_cloud as eii
from eii_cloud.evidence.graph import EvidenceGraph
from eii_cloud.experiments import make_reference_scenario
from eii_cloud.visualization import DetectorComparisonChart, EventSequenceChart, EvidenceGraphChart, MissingnessTimeline

out = Path(sys.argv[1] if len(sys.argv) > 1 else "reference_chart_out")
sc = make_reference_scenario(seed=0)

engine = eii.EIIEngine(
    envelopes=sc.envelopes,
    detectors=[
        {"type": "threshold", "parameter": "gpu_temp", "envelope": "gpu_temp_static", "detector_id": "threshold:gpu_temp"},
        {"type": "missingness", "parameter": "gpu_temp", "detector_id": "missing:gpu_temp"},
        {"type": "flatline", "parameter": "gpu_temp", "min_run": 8, "tolerance": 0.0, "detector_id": "flatline:gpu_temp"},
        {"type": "context_contradiction", "parameter": "gpu_temp", "detector_id": "context:gpu_temp"},
    ],
    rules=sc.rules,
    context=sc.context,
)
result = engine.run(sc.dataset, reference=sc.reference)
print(result.summary())
print("ground truth:", sc.ground_truth)

em = eii.ExportManager(out, dpi=300)
for mode in ("annotated", "clean", "flags_only"):
    chart = eii.EIICloudChart(sc.dataset, result, "gpu_temp", envelope="gpu_temp_static", mode=mode)
    fig = chart.render()
    files = em.export(fig, f"gpu_temp_{mode}", metadata={"analysis_id": result.analysis_id, "mode": mode})
    em.write_manifest(f"gpu_temp_{mode}", files, {"annotation_ids": chart.annotation_ids()})
    plt.close(fig)
eii.write_annotations_json(result, out / "gpu_temp_annotations.json", sc.dataset, "gpu_temp")
result.to_json(out / "result.json")

for name, fig in (("missingness", MissingnessTimeline(sc.dataset, result).render()),
                  ("events", EventSequenceChart(sc.dataset, result).render()),
                  ("comparison", DetectorComparisonChart(sc.dataset, result).render())):
    em.export(fig, name, formats=("svg", "png"))
    plt.close(fig)
for f in result.findings:
    if f.rule_id:
        fig = EvidenceGraphChart(EvidenceGraph.from_finding(f)).render()
        em.export(fig, f"evidence_{f.rule_id}", formats=("svg", "png"))
        plt.close(fig)
print(f"written to {out.resolve()}")

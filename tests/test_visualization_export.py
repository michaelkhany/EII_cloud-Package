import json

import matplotlib.pyplot as plt
import pytest

from eii_cloud import (
    ChartStyle,
    EIICloudChart,
    EvidenceGraphChart,
    ExportManager,
    Layers,
    annotations_document,
    build_annotations,
)
from eii_cloud.evidence.graph import EvidenceGraph
from eii_cloud.experiments import exp05_export_reproducibility
from eii_cloud.export import svg_ids
from eii_cloud.visualization import (
    AnomalyScoreChart,
    DetectorComparisonChart,
    EventSequenceChart,
    MissingnessTimeline,
    MultiSignalChart,
    get_chart,
    list_charts,
)


def test_annotations_json_is_pure(scenario, engine_and_result):
    _, result = engine_and_result
    doc = annotations_document(result, scenario.dataset, "gpu_temp")
    assert doc["n_annotations"] == 6
    ids = [a["annotation_id"] for a in doc["annotations"]]
    assert len(set(ids)) == 6 and all(i.startswith("ann-") for i in ids)
    assert doc["missing_regions"] == [{"finding_id": doc["missing_regions"][0]["finding_id"], "parameter_ids": ["gpu_temp"],
                                       "start_time": 50, "end_time": 56}]
    a = doc["annotations"][0]
    for key in ("analysis_id", "asset_id", "category", "severity", "start_time", "end_time", "anchor_time", "text",
                "rule_id", "detector_id", "evidence", "context", "versions"):
        assert key in a
    json.dumps(doc)  # serialisable


@pytest.mark.parametrize("mode", ["annotated", "clean", "flags_only"])
def test_modes_render_and_keep_ids(tmp_path, scenario, engine_and_result, mode):
    _, result = engine_and_result
    ch = EIICloudChart(scenario.dataset, result, "gpu_temp", envelope="gpu_temp_static", mode=mode)
    fig = ch.render()
    files = ExportManager(tmp_path, dpi=50).export(fig, f"c_{mode}", formats=("svg", "pdf", "tiff", "png"))
    plt.close(fig)
    assert all(p.exists() and p.stat().st_size > 0 for p in files.values())
    ids = svg_ids(files["svg"])
    assert "eii-observed" in ids and "eii-cloud" in ids
    missing_ids = {i for i in ids if i.startswith("eii-missing-")}
    assert missing_ids == {"eii-missing-50-56"}
    if mode == "clean":
        assert not any(i.startswith("flag-") for i in ids)
    else:
        assert {f"flag-{a}" for a in ch.annotation_ids()} <= ids
    if mode == "annotated":
        assert set(ch.annotation_ids()) <= ids


def test_rendering_does_not_alter_result(scenario, engine_and_result):
    _, result = engine_and_result
    before = json.dumps(result.to_dict(), sort_keys=True, default=str)
    for mode in ("annotated", "clean", "flags_only"):
        plt.close(EIICloudChart(scenario.dataset, result, "gpu_temp", mode=mode).render())
    assert json.dumps(result.to_dict(), sort_keys=True, default=str) == before


def test_layers_and_panel(scenario, engine_and_result):
    _, result = engine_and_result
    layers = Layers.for_mode("annotated", howto=False, legend=False, context=True)
    ch = EIICloudChart(scenario.dataset, result, "gpu_temp", layers=layers, context_column="gpu_util",
                       style=ChartStyle(figsize=(12, 6), dpi=60), annotation_panel=True)
    fig = ch.render()
    assert len(fig.axes) >= 2
    plt.close(fig)
    with pytest.raises(ValueError):
        EIICloudChart(scenario.dataset, result, "gpu_temp", mode="bogus")


def test_no_interpolation_across_gaps(scenario, engine_and_result):
    import numpy as np
    _, result = engine_and_result
    ch = EIICloudChart(scenario.dataset, result, "gpu_temp", mode="clean")
    fig = ch.render()
    ydata = ch.artists["observed"].get_ydata()
    assert np.isnan(np.asarray(ydata, dtype=float)[50:57]).all()
    plt.close(fig)


def test_auxiliary_charts(scenario, engine_and_result):
    _, result = engine_and_result
    ds = scenario.dataset
    figs = [MissingnessTimeline(ds, result).render(), EventSequenceChart(ds, result).render(),
            MultiSignalChart(ds, result).render(), DetectorComparisonChart(ds, result).render(),
            AnomalyScoreChart(ds, result, "threshold:gpu_temp").render(),
            EvidenceGraphChart(EvidenceGraph.from_finding(result.findings[0])).render()]
    for f in figs:
        plt.close(f)
    assert "EIICloudChart" in list_charts() and get_chart("EvidenceGraph") is EvidenceGraphChart


def test_exp05():
    rep = exp05_export_reproducibility(0)
    assert rep["annotation_ids_identical_across_modes"] and rep["svg_contains_annotation_ids"]
    assert rep["json_unchanged_by_rendering"] and rep["findings_unchanged_by_rerun"]


def test_annotation_numbering_time_ordered(scenario, engine_and_result):
    _, result = engine_and_result
    anns = build_annotations(result, scenario.dataset, "gpu_temp")
    assert [a.number for a in anns] == list(range(1, len(anns) + 1))
    assert [a.start_time for a in anns] == sorted(a.start_time for a in anns)

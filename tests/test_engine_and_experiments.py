import json

import numpy as np
import pytest

from eii_cloud import AnalysisResult, EIIEngine, StreamingEngine, findings_equal, split_chunks
from eii_cloud.evaluation import bootstrap_ci, classification_metrics, event_metrics
from eii_cloud.evidence.graph import EvidenceGraph
from eii_cloud.experiments import (
    ExperimentRunner,
    exp00_reference_reproduction,
    exp01_context_vs_static,
    exp03_observability_degradation,
    exp06_batch_vs_streaming,
    inject,
)


def test_exp00_exact_intervals(scenario, engine_and_result):
    _, result = engine_and_result
    got = {c: [(f.start_index, f.end_index) for f in result.findings if f.category.value == c and f.rule_id is None]
           for c in scenario.ground_truth}
    assert got == scenario.ground_truth
    rep = exp00_reference_reproduction(0)
    assert rep["all_intervals_exact"]


def test_result_json_roundtrip(tmp_path, engine_and_result):
    _, result = engine_and_result
    p = tmp_path / "r.json"
    result.to_json(p)
    back = AnalysisResult.from_json(p)
    assert findings_equal(back.findings, result.findings)
    assert back.analysis_id == result.analysis_id
    doc = json.loads(p.read_text())
    assert doc["schema_version"] == "1.0"
    for f in doc["findings"]:
        for key in ("finding_id", "asset_id", "parameter_ids", "start_time", "end_time", "category", "detector_id",
                    "severity", "truth_status", "evidence", "context_snapshot", "description", "provenance"):
            assert key in f


def test_deterministic_ids_and_config_hash(scenario, engine_and_result):
    engine, result = engine_and_result
    again = engine.run(scenario.dataset, scenario.reference)
    assert findings_equal(result.findings, again.findings)
    assert again.analysis_id == result.analysis_id


def test_streaming_equals_batch(scenario, engine_and_result):
    engine, batch = engine_and_result
    stream = StreamingEngine(engine, scenario.reference)
    for chunk in split_chunks(scenario.dataset, 7):
        stream.push(chunk)
    final = stream.flush()
    assert findings_equal(batch.findings, final.findings)
    assert all(not f.provisional for f in final.findings)
    rep = exp06_batch_vs_streaming(0, chunk_size=10)
    assert rep["identical_after_flush"] and rep["finalised_findings_never_changed"]


def test_streaming_late_event_changes_provisional_only(scenario, engine_and_result):
    engine, _ = engine_and_result
    ds = scenario.dataset
    from eii_cloud import EventLog
    quiet = EventLog(ds.events.frame.iloc[0:0], ds.events.coverage)  # drop the xid event
    from eii_cloud import EIIDataset
    ds2 = EIIDataset(ds.time.index, ds.values, ds.context, ds.expected, units=ds.units, events=quiet,
                     asset_id=ds.asset_id, name=ds.name)
    stream = StreamingEngine(engine, scenario.reference)
    stream.push(ds2.iloc(0, 50))
    assert not any(f.rule_id == "deviation_then_gpu_error" for f in stream.final.values())
    res = stream.push_events([{"event_id": "xid-001", "event_type": "gpu_xid", "timestamp": 44, "asset_id": "gpu0",
                               "source": "gpu_xid"}])
    assert any(f.rule_id == "deviation_then_gpu_error" for f in res.findings)


def test_non_causal_engine_warns(scenario):
    eng = EIIEngine(envelopes=scenario.envelopes, detectors=[{"type": "change_point", "parameter": "gpu_temp"}])
    with pytest.warns(RuntimeWarning):
        StreamingEngine(eng)


def test_experiment_runner_reproduce(tmp_path):
    cfg = {"name": "t", "dataset": {"synthetic": {"scenario": "reference", "seed": 3}},
           "charts": [{"parameter": "gpu_temp", "envelope": "gpu_temp_static", "modes": ["clean"]}],
           "export": {"formats": ["svg"], "dpi": 50}}
    out = tmp_path / "run"
    result = ExperimentRunner(cfg, out).run()
    assert (out / "manifest.json").exists() and (out / "charts" / "gpu_temp_clean.svg").exists()
    rep = ExperimentRunner.reproduce(out)
    assert rep["identical"] and rep["config_sha256_match"]
    assert len(result.findings) == json.loads((out / "manifest.json").read_text())["n_findings"]


def test_injection_and_exp03(scenario):
    ds = inject(scenario.dataset, "gpu_temp", "missing", 10, 12)
    assert np.isnan(ds.value("gpu_temp")[10:13]).all()
    ds = inject(scenario.dataset, "gpu_temp", "not_expected", 10, 12)
    assert not ds.expected_mask("gpu_temp")[10:13].any()
    rep = exp03_observability_degradation(0)
    assert rep["monitoring_faults_flagged_as_physical"] == 0
    assert rep["monitoring_faults_recovered"] == 2
    assert rep["scheduled_gap_flagged_as_missing"] is False


def test_exp01_context_beats_static():
    rep = exp01_context_vs_static(0)
    v = rep["variants"]
    assert v["static_threshold"]["event_recall"] < v["context_plus_composite_rules"]["event_recall"]
    assert v["context_plus_composite_rules"]["false_alert_episodes"] == 0
    assert v["context_conditioned"]["coverage"]["by_context"]["high_load"]["coverage"] == 0.0


def test_metrics():
    from eii_cloud import Category, Finding
    f = Finding("a", ["x"], Category.VALUE_DEVIATION, 10, 12, 10, 12, "d")
    g = Finding("a", ["x"], Category.VALUE_DEVIATION, 40, 41, 40, 41, "d")
    m = event_metrics([f, g], [(8, 12), (20, 25)], 100, truth_series=[-1] * 10 + [0] * 90)
    assert m["event_recall"] == 0.5 and m["false_alert_episodes"] == 1 and m["abstention_rate"] == 0.1
    assert m["mean_detection_delay"] == 2.0
    cm = classification_metrics(["a", "a", "b"], ["a", "b", "b"])
    assert cm["balanced_accuracy"] == 0.75
    ci = bootstrap_ci([1, 2, 3, 4], lambda u: float(np.mean(u)), n_boot=200)
    assert ci["low"] <= ci["estimate"] <= ci["high"]


def test_evidence_graph(engine_and_result):
    _, result = engine_and_result
    f = next(f for f in result.findings if f.rule_id == "instability_under_load")
    g = EvidenceGraph.from_finding(f)
    kinds = {n.kind for n in g.nodes}
    assert {"finding", "and", "persists_for", "signal", "outside_envelope"} <= kinds
    assert len(g.edges) == len(g.nodes) - 1

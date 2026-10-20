import numpy as np
import pytest

from eii_cloud import (
    BaseDetector,
    Category,
    ChangePointDetector,
    ContextDefinition,
    CrossSignalConstraintDetector,
    CUSUMDetector,
    EIIDataset,
    EWMADetector,
    FlatlineDetector,
    MissingnessDetector,
    RateOfChangeDetector,
    RobustDeviationDetector,
    ThresholdDetector,
    TruthSeries,
    build_detector,
    fixed_bounds,
    get_detector,
    list_detectors,
    register_detector,
)


def test_threshold_unknown_when_missing_or_unsupported():
    ds = EIIDataset(time=range(5), values={"x": [1.0, 9.0, np.nan, 5.0, 9.0]})
    env = fixed_bounds(ds, "x", 0, 8)
    env.support[4] = False
    out = ThresholdDetector("x", envelope="e").detect(ds, {"e": env})
    assert out.truth.values.tolist() == [0, 1, -1, 0, -1]
    assert len(out.findings) == 1 and out.findings[0].evidence["direction"] == "above"
    assert out.findings[0].evidence["magnitude"] == 1.0


def test_threshold_static_limits():
    ds = EIIDataset(time=range(3), values={"x": [1.0, 50.0, 2.0]})
    out = ThresholdDetector("x", upper=10).detect(ds)
    assert [(f.start_index, f.end_index) for f in out.findings] == [(1, 1)]


def test_missingness_scheduled_gap_not_a_finding():
    ds = EIIDataset(time=range(6), values={"x": [1, np.nan, np.nan, 4, np.nan, 6]},
                    expected={"x": [True, True, True, True, False, True]})
    out = MissingnessDetector("x").detect(ds)
    assert [(f.start_index, f.end_index) for f in out.findings] == [(1, 2)]
    assert out.findings[0].evidence["last_observed_before"] == 0
    assert out.findings[0].evidence["first_observed_after"] == 3


def test_missingness_time_axis_gap():
    ds = EIIDataset(time=[0, 1, 2, 10, 11], values={"x": [1, 2, 3, 4, 5]})
    out = MissingnessDetector("x", max_gap=2).detect(ds)
    assert [(f.start_index, f.end_index) for f in out.findings] == [(3, 3)]
    assert out.findings[0].evidence.get("time_axis_gap") is True


def test_flatline_tolerance_and_regime_restriction():
    x = np.array([1, 2, 3, 3, 3, 3, 3, 3.004, 4, 5, 6], dtype=float)
    ds = EIIDataset(time=range(11), values={"x": x}, context={"u": [0] * 4 + [100] * 7})
    assert FlatlineDetector("x", min_run=5, tolerance=0.0).detect(ds).findings[0].n_samples == 5
    assert FlatlineDetector("x", min_run=6, tolerance=0.01).detect(ds).findings[0].n_samples == 6
    cd = ContextDefinition().add_regime("busy", "u", ">=", 50)
    out = FlatlineDetector("x", min_run=5, tolerance=0.0, regimes=["busy"]).detect(ds, context=cd)
    assert [(f.start_index, f.end_index) for f in out.findings] == [(4, 6)]


def test_rate_of_change():
    ds = EIIDataset(time=[0, 1, 2, 4], values={"x": [0.0, 1.0, 10.0, 12.0]})
    out = RateOfChangeDetector("x", max_rate=5).detect(ds)
    assert out.truth.values.tolist() == [-1, 0, 1, 0]


def test_ewma_and_cusum_detect_shift():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 200)
    x[120:] += 2.5
    ds = EIIDataset(time=range(200), values={"x": x})
    ref = EIIDataset(time=range(100), values={"x": x[:100]})
    for det in (EWMADetector("x", lam=0.2, L=3), CUSUMDetector("x", k=0.5, h=5)):
        det.fit(ref)
        out = det.detect(ds)
        assert out.findings, type(det).__name__
        assert out.findings[0].start_index >= 120 and out.findings[0].start_index < 135
        assert out.findings[0].category == Category.TEMPORAL_INSTABILITY


def test_robust_deviation_scores_outlier():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 80)
    x[50] = 15
    ds = EIIDataset(time=range(80), values={"x": x})
    out = RobustDeviationDetector("x", window=20, k=5).detect(ds)
    assert [(f.start_index, f.end_index) for f in out.findings] == [(50, 50)]


def test_change_point_binary_segmentation():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(0, 0.5, 50), rng.normal(5, 0.5, 50), rng.normal(0, 0.5, 50)])
    ds = EIIDataset(time=range(150), values={"x": x})
    det = ChangePointDetector("x", min_size=5)
    out = det.detect(ds)
    cps = sorted(f.start_index for f in out.findings)
    assert len(cps) == 2 and abs(cps[0] - 50) <= 2 and abs(cps[1] - 100) <= 2
    assert det.causal is False


def test_cross_signal_constraint():
    ds = EIIDataset(time=range(4), values={"power": [10.0, 50.0, 400.0, np.nan], "util": [0.0, 10.0, 10.0, 10.0]})
    det = CrossSignalConstraintDetector("power", lhs="power", op="<=", rhs={"util": 3.0, "const": 30}, tolerance=1)
    out = det.detect(ds)
    assert out.truth.values.tolist() == [0, 0, 1, -1]
    assert out.findings[0].parameter_ids == ["power", "util"]


def test_registry_and_external_detector():
    @register_detector("test_spike")
    class Spike(BaseDetector):
        category = Category.TEMPORAL_INSTABILITY

        def __init__(self, parameter, jump=5.0, **kw):
            super().__init__(parameter, jump=jump, **kw)
            self.jump = jump

        def evaluate(self, dataset, envelopes, context):
            x = dataset.value(self.parameter)
            d = np.abs(np.diff(x, prepend=x[0]))
            return TruthSeries.from_bool(d > self.jump, np.isnan(d)), d

    assert "test_spike" in list_detectors()
    det = build_detector({"type": "test_spike", "parameter": "x", "jump": 2})
    ds = EIIDataset(time=range(4), values={"x": [0.0, 0.5, 10.0, 10.2]})
    out = det.detect(ds)
    assert [(f.start_index, f.end_index) for f in out.findings] == [(2, 2)]
    with pytest.raises(KeyError):
        get_detector("does_not_exist")

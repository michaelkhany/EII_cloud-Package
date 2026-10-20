import numpy as np
import pandas as pd
import pytest

from eii_cloud import (
    EIIDataset,
    EventLog,
    SeasonalNaiveForecaster,
    build_envelope,
    contextual_quantiles,
    fixed_bounds,
    forecast_interval,
    physical_constraints,
    regime_lookup,
    regression_residual,
    rolling_robust,
    scheduled_bounds,
)
from eii_cloud.context import ContextDefinition


def test_time_axis_requires_sorted_unique():
    with pytest.raises(ValueError):
        EIIDataset(time=[0, 2, 1], values={"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        EIIDataset(time=[0, 1, 1], values={"x": [1, 2, 3]})


def test_datetime_axis_offsets_and_labels():
    idx = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    ds = EIIDataset(time=idx, values={"x": [1, 2, 3, 4]})
    assert ds.time.is_datetime
    assert ds.time.offset("10min") == 600.0
    assert ds.time.label(1) == "2026-01-01T00:05:00+00:00"
    assert ds.time.from_label(ds.time.label(2)) == ds.time.t[2]


def test_missing_when_expected_respects_schedule():
    ds = EIIDataset(time=range(4), values={"x": [1, np.nan, np.nan, 4]}, expected={"x": [True, True, False, True]})
    assert ds.missing_when_expected("x").tolist() == [False, True, False, False]


def test_save_load_roundtrip(tmp_path, scenario):
    d = scenario.dataset.save(tmp_path / "ds")
    back = EIIDataset.load(d)
    assert back.content_hash() == scenario.dataset.content_hash()
    assert back.parameters == scenario.dataset.parameters
    assert len(back.events) == 1


def test_fixed_and_scheduled_bounds():
    ds = EIIDataset(time=range(10), values={"x": np.linspace(0, 9, 10)})
    env = fixed_bounds(ds, "x", 2, 7)
    assert env.contains(ds.value("x")).values.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    sched = scheduled_bounds(ds, "x", [{"start": 0, "end": 4, "lower": 0, "upper": 10}])
    assert sched.support.tolist() == [True] * 5 + [False] * 5
    # outside the schedule the containment is UNKNOWN, not FALSE
    assert sched.contains(ds.value("x")).values.tolist()[5:] == [-1] * 5


def test_rolling_robust_is_causal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 60)
    ds = EIIDataset(time=range(60), values={"x": x})
    env = rolling_robust(ds, "x", window=10, k=3)
    assert not env.support[:5].any()
    # changing a future value must not change the envelope at earlier samples
    x2 = x.copy()
    x2[40] = 100
    env2 = rolling_robust(EIIDataset(time=range(60), values={"x": x2}), "x", window=10, k=3)
    assert np.allclose(env.upper[:41], env2.upper[:41], equal_nan=True)
    assert not np.allclose(env.upper[41:51], env2.upper[41:51], equal_nan=True)


def test_contextual_quantiles_reports_insufficient_support():
    rng = np.random.default_rng(1)
    ref = EIIDataset(time=range(200), values={"x": rng.normal(0, 1, 200)},
                     context={"mode": ["a"] * 190 + ["b"] * 10})
    ds = EIIDataset(time=range(3), values={"x": [0.0, 0.0, 0.0]}, context={"mode": ["a", "b", "c"]})
    env = contextual_quantiles(ds, "x", ref, "mode", min_support=30)
    assert env.support.tolist() == [True, False, False]
    assert any("b" in w for w in env.warnings) and any("c" in w for w in env.warnings)
    assert env.contains(ds.value("x")).values.tolist() == [1, -1, -1]


def test_regime_lookup_and_regression_residual():
    rng = np.random.default_rng(2)
    util = rng.uniform(0, 100, 300)
    temp = 40 + 0.2 * util + rng.normal(0, 0.5, 300)
    ref = EIIDataset(time=range(300), values={"t": temp}, context={"u": util})
    ds = EIIDataset(time=range(3), values={"t": [40.0, 60.0, 70.0]}, context={"u": [0.0, 100.0, 100.0]})
    env = regression_residual(ds, "t", ref, ["u"], q=(0.01, 0.99))
    assert env.contains(ds.value("t")).values.tolist() == [1, 1, 0]
    cd = ContextDefinition().add_regime("hot", "u", ">=", 50)
    lk = regime_lookup(ds, "t", {"hot": {"lower": 55, "upper": 65}, "default": {"lower": 35, "upper": 45}}, cd)
    assert lk.contains(ds.value("t")).values.tolist() == [1, 1, 0]


def test_physical_constraints_linear_expression():
    ds = EIIDataset(time=range(3), values={"p": [10.0, 20.0, 50.0], "u": [0.0, 50.0, 100.0]})
    env = physical_constraints(ds, "p", upper={"u": 0.3, "const": 5})
    assert env.contains(ds.value("p")).values.tolist() == [0, 1, 0]


def test_forecast_interval_plugin():
    x = np.tile(np.arange(10, dtype=float), 6)
    ds = EIIDataset(time=range(60), values={"x": x})
    env = forecast_interval(ds, "x", SeasonalNaiveForecaster(period=10, k=3))
    assert not env.support[:10].any() and env.support[10:].all()
    assert env.contains(ds.value("x")).values.tolist()[10:] == [1] * 50


def test_build_envelope_from_spec_external_aligned(scenario):
    ds = scenario.dataset.iloc(20, 60)
    env = build_envelope(ds, scenario.envelopes["gpu_temp_static"])
    assert env.n == 40 and env.support.all()
    assert env.coverage(ds.value("gpu_temp"))["n_supported_observed"] == 33  # 7 missing samples excluded


def test_coverage_by_context(scenario):
    ds = scenario.dataset
    env = build_envelope(ds, scenario.envelopes["gpu_temp_static"])
    cov = env.coverage(ds.value("gpu_temp"), scenario.context.regime_labels(ds))
    assert set(cov["by_context"]) == {"default", "high_load"}


def test_event_log_coverage_and_dedupe():
    ev = EventLog.from_records([
        {"event_id": "a", "event_type": "err", "timestamp": 10},
        {"event_id": "b", "event_type": "err", "timestamp": 10.4},
        {"event_id": "c", "event_type": "err", "timestamp": 30},
    ])
    assert len(ev.deduplicate(tolerance=1)) == 2
    with pytest.raises(ValueError):
        EventLog.from_records([{"event_id": "a", "event_type": "e", "timestamp": 1},
                               {"event_id": "a", "event_type": "e", "timestamp": 2}])

import numpy as np
import pytest

from eii_cloud.rules._windows import run_level, window_all, window_any, window_count
from eii_cloud.truth import UNKNOWN, TruthSeries, runs_of


def test_kleene_tables():
    a = TruthSeries([1, 1, 1, 0, 0, 0, -1, -1, -1])
    b = TruthSeries([1, 0, -1, 1, 0, -1, 1, 0, -1])
    assert (a & b).values.tolist() == [1, 0, -1, 0, 0, 0, -1, 0, -1]
    assert (a | b).values.tolist() == [1, 1, 1, 1, 0, -1, 1, -1, -1]
    assert (~a).values.tolist() == [0, 0, 0, 1, 1, 1, -1, -1, -1]


def test_negation_never_resolves_unknown():
    u = TruthSeries.constant(5, UNKNOWN)
    assert (~u).is_unknown.all()
    assert (~~u).is_unknown.all()


def test_runs():
    assert runs_of(np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)) == [(1, 2), (4, 4), (7, 7)]
    assert TruthSeries([1, 1, 0, -1, 1]).runs() == [(0, 1), (4, 4)]


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        TruthSeries([2, 0])


def test_window_any_semantics():
    t = np.arange(6, dtype=float)
    a = TruthSeries([0, 0, 1, 0, 0, 0])
    # look back 1: TRUE at 2 and 3; sample 0 window incomplete -> UNKNOWN
    assert window_any(a, t, 1, 0).values.tolist() == [-1, 0, 1, 1, 0, 0]
    # look forward 2: TRUE at 0,1,2; samples 4,5 incomplete -> UNKNOWN
    assert window_any(a, t, 0, 2).values.tolist() == [1, 1, 1, 0, -1, -1]
    # unknown inside window propagates when no TRUE
    b = TruthSeries([0, -1, 0, 0, 0, 0])
    assert window_any(b, t, 1, 0).values.tolist() == [-1, -1, -1, 0, 0, 0]


def test_window_all_persistence():
    t = np.arange(6, dtype=float)
    a = TruthSeries([1, 1, 1, 0, 1, 1])
    assert window_all(a, t, 2).values.tolist() == [-1, -1, 1, 0, 0, 0]
    u = TruthSeries([1, -1, 1, 1, 1, 1])
    assert window_all(u, t, 2).values.tolist() == [-1, -1, -1, -1, 1, 1]


def test_window_count_onsets():
    t = np.arange(8, dtype=float)
    a = TruthSeries([1, 0, 1, 1, 0, 1, 0, 0])
    ts, count = window_count(a, t, 4, 2)
    assert count.tolist() == [1, 1, 2, 2, 2, 2, 2, 1]
    assert ts.values.tolist() == [-1, -1, 1, 1, 1, 1, 1, 0]


def test_run_level_during_and_overlaps():
    a = TruthSeries([0, 1, 1, 0, 1, 1, 0])
    b = TruthSeries([1, 1, 1, 1, 1, 0, 0])
    assert run_level(a, b, "during").values.tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert run_level(a, b, "overlaps").values.tolist() == [0, 1, 1, 0, 1, 1, 0]
    c = TruthSeries([1, 1, -1, 1, 1, 1, 1])
    assert run_level(a, c, "during").values.tolist() == [0, -1, -1, 0, 1, 1, 0]

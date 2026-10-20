import numpy as np
import pytest
import yaml

from eii_cloud import C, EIIDataset, EventLog, Rule, RuleSet
from eii_cloud.context import ContextDefinition
from eii_cloud.rules import (
    EvalContext,
    RuleCompileError,
    RuleCycleError,
    compile_expression,
    validate_rule_document,
)
from eii_cloud.rules.nodes import RuleError


def _ds():
    ev = EventLog.from_records(
        [{"event_id": "e1", "event_type": "err", "timestamp": 6}],
        coverage=[{"source": "err", "start": 0, "end": 7}])
    return EIIDataset(time=range(10), values={"x": [0, 0, 5, 5, 5, 0, 0, np.nan, 0, 0]},
                      context={"load": [0, 0, 90, 90, 90, 90, 0, 0, 0, 0]}, events=ev)


def test_signal_condition_three_valued():
    ctx = EvalContext(_ds())
    assert C.signal("x", ">", 1).evaluate(ctx).values.tolist() == [0, 0, 1, 1, 1, 0, 0, -1, 0, 0]


def test_event_coverage_makes_absence_unknown():
    ctx = EvalContext(_ds())
    ev = C.event("err").evaluate(ctx)
    assert ev.values.tolist() == [0, 0, 0, 0, 0, 0, 1, 0, -1, -1]
    # "no error" is not evidence where the collector was down
    assert (~C.event("err")).evaluate(ctx).values.tolist()[8:] == [-1, -1]


def test_before_after_within_semantics():
    ctx = EvalContext(_ds())
    high = C.signal("x", ">", 1)
    err = C.event("err")
    assert high.before(err, 3).evaluate(ctx).values.tolist() == [0, 0, 0, 1, 1, 0, 0, -1, 0, 0]
    assert err.after(high, 2).evaluate(ctx).values.tolist() == [0, 0, 0, 0, 0, 0, 1, 0, -1, -1]
    assert high.within(err, 2).evaluate(ctx).values.tolist() == [0, 0, 0, 0, 1, 0, 0, -1, 0, 0]


def test_persists_for_and_count_within():
    ctx = EvalContext(_ds())
    load = C.signal("load", ">=", 50)
    # any FALSE inside the window makes the result FALSE even where the window is incomplete
    assert load.persists_for(2).evaluate(ctx).values.tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    assert C.signal("x", ">", 1).count_within(9, 1).evaluate(ctx).values.tolist()[-1] == 1


def test_rule_dependencies_order_and_cycle():
    rs = RuleSet([
        Rule("b", {"and": [{"rule": "a"}, {"signal": "x", "op": ">", "value": 1}]}),
        Rule("a", {"signal": "load", "op": ">=", "value": 50}),
    ])
    assert [r.id for r in rs.ordered()] == ["a", "b"]
    out = rs.evaluate(EvalContext(_ds()))
    assert [(f.start_index, f.end_index) for f in out["b"].findings] == [(2, 4)]
    cyc = RuleSet([Rule("p", {"rule": "q"}), Rule("q", {"not": {"rule": "p"}})])
    with pytest.raises(RuleCycleError):
        cyc.ordered()
    with pytest.raises(RuleError):
        RuleSet([Rule("p", {"rule": "missing"})]).ordered()


def test_compiler_rejects_unknown_operators_and_runs_no_code():
    with pytest.raises(RuleCompileError):
        compile_expression({"exec": "import os"})
    with pytest.raises(RuleCompileError):
        compile_expression({"and": []})
    with pytest.raises(RuleCompileError):
        compile_expression({"signal": "x", "op": ">"})
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load("a: !!python/object/apply:os.system ['echo hi']")
    with pytest.raises(RuleCompileError):
        validate_rule_document({"rules": [{"id": "a", "expression": {"signal": "x", "op": ">", "value": 1}},
                                          {"id": "a", "expression": {"signal": "x", "op": ">", "value": 1}}]})


def test_yaml_roundtrip(tmp_path, scenario):
    path = tmp_path / "rules.yaml"
    scenario.rules.to_yaml(path)
    back = RuleSet.from_yaml(path)
    assert back.to_dict() == scenario.rules.to_dict()


def test_contradicts_context_and_settle():
    ds = _ds()
    cd = ContextDefinition().add_regime("busy", "load", ">=", 50)
    cd.expect(regime="busy", parameter="x", kind="min_level", value=3, settle=1)
    ctx = EvalContext(ds, context=cd)
    # busy at 2..5; settle skips 2; x=5 at 3,4 satisfies; x=0 at 5 violates
    assert C.contradicts_context("x").evaluate(ctx).values.tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]


def test_rule_finding_carries_evidence(scenario, engine_and_result):
    _, result = engine_and_result
    f = next(f for f in result.findings if f.rule_id == "deviation_then_gpu_error")
    assert f.evidence["matched_event_ids"] == ["xid-001"]
    assert f.evidence["expression"]["kind"] == "before"
    assert f.reference_id == "rule:deviation_then_gpu_error@1"
    assert f.provenance["rule"]["expression"]["before"]["window"] == 12


def test_unknown_evidence_yields_no_finding(engine_and_result):
    _, result = engine_and_result
    # missing telemetry AND NOT error: the log collector was down 50..56 -> UNKNOWN, must abstain
    assert not [f for f in result.findings if f.rule_id == "error_without_telemetry"]
    ts = np.asarray(result.truth_series["rule:error_without_telemetry"])
    assert (ts[50:57] == -1).all()

"""Compile serialisable rule specifications (YAML/JSON/dict) into node trees.

The compiler maps a closed vocabulary of keys to node classes.  It never evaluates
strings as code, so rule files from untrusted sources cannot execute anything.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .nodes import (
    After,
    And,
    Before,
    Context,
    ContradictsContext,
    CountWithin,
    During,
    Event,
    FindingPresent,
    MissingWhenExpected,
    Node,
    Not,
    Or,
    OutsideEnvelope,
    Overlaps,
    PersistsFor,
    RuleError,
    RuleRef,
    Signal,
    Within,
)

RULE_SCHEMA_VERSION = "1.0"


class RuleCompileError(RuleError):
    pass


def _require(spec: Mapping[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in spec]
    if missing:
        raise RuleCompileError(f"{where}: missing keys {missing}")


def compile_expression(spec: Any) -> Node:
    """Turn a nested dict into a :class:`Node` tree."""
    if isinstance(spec, Node):
        return spec
    if not isinstance(spec, Mapping):
        raise RuleCompileError(f"expression must be a mapping, got {type(spec).__name__}")
    if len(spec) == 0:
        raise RuleCompileError("empty expression")
    # leaf: {signal: name, op: .., value: ..}
    if "signal" in spec:
        _require(spec, ("op", "value"), "signal")
        extra = set(spec) - {"signal", "op", "value"}
        if extra:
            raise RuleCompileError(f"signal: unexpected keys {sorted(extra)}")
        return Signal(str(spec["signal"]), str(spec["op"]), spec["value"])
    if len(spec) != 1:
        raise RuleCompileError(f"expression must have exactly one operator key, got {sorted(spec)}")
    (key, val), = spec.items()
    if key == "outside_envelope":
        if isinstance(val, str):
            return OutsideEnvelope(val)
        _require(val, ("parameter",), key)
        return OutsideEnvelope(str(val["parameter"]), val.get("envelope"))
    if key == "missing_when_expected":
        return MissingWhenExpected(str(val if isinstance(val, str) else val["parameter"]))
    if key == "event":
        if isinstance(val, str):
            return Event(val)
        _require(val, ("event_type",), key)
        return Event(str(val["event_type"]), val.get("source"), val.get("tolerance"))
    if key == "context":
        return Context(str(val if isinstance(val, str) else val["regime"]))
    if key == "finding":
        if isinstance(val, str):
            return FindingPresent(detector=val)
        return FindingPresent(val.get("detector"), val.get("category"), val.get("parameter"))
    if key == "rule":
        return RuleRef(str(val))
    if key == "contradicts_context":
        if isinstance(val, str):
            return ContradictsContext(val)
        _require(val, ("parameter",), key)
        return ContradictsContext(str(val["parameter"]), val.get("expectation"))
    if key == "and":
        return And([compile_expression(v) for v in _as_list(val, key)])
    if key == "or":
        return Or([compile_expression(v) for v in _as_list(val, key)])
    if key == "not":
        return Not(compile_expression(val))
    if key == "before":
        _require(val, ("condition", "next", "window"), key)
        return Before(compile_expression(val["condition"]), compile_expression(val["next"]), val["window"])
    if key == "after":
        _require(val, ("condition", "prior", "window"), key)
        return After(compile_expression(val["condition"]), compile_expression(val["prior"]), val["window"])
    if key == "within":
        _require(val, ("condition", "other", "window"), key)
        return Within(compile_expression(val["condition"]), compile_expression(val["other"]), val["window"])
    if key == "during":
        _require(val, ("condition", "interval"), key)
        return During(compile_expression(val["condition"]), compile_expression(val["interval"]))
    if key == "overlaps":
        _require(val, ("condition", "interval"), key)
        return Overlaps(compile_expression(val["condition"]), compile_expression(val["interval"]))
    if key == "persists_for":
        _require(val, ("condition", "window"), key)
        return PersistsFor(compile_expression(val["condition"]), val["window"])
    if key == "count_within":
        _require(val, ("condition", "window", "min_count"), key)
        return CountWithin(compile_expression(val["condition"]), val["window"], int(val["min_count"]))
    raise RuleCompileError(f"unknown operator {key!r}")


def _as_list(val: Any, key: str) -> list:
    if not isinstance(val, (list, tuple)) or not val:
        raise RuleCompileError(f"{key}: expects a non-empty list")
    return list(val)


def load_rule_file(path: str | Path) -> dict[str, Any]:
    """Safe-load a YAML/JSON rule file and validate the top-level structure."""
    text = Path(path).read_text()
    data = yaml.safe_load(text)
    validate_rule_document(data)
    return data


def validate_rule_document(data: Any) -> None:
    if not isinstance(data, Mapping):
        raise RuleCompileError("rule document must be a mapping with a 'rules' list")
    if "rules" not in data or not isinstance(data["rules"], list):
        raise RuleCompileError("rule document needs a 'rules' list")
    ids = [r.get("id") for r in data["rules"]]
    if any(i is None for i in ids):
        raise RuleCompileError("every rule needs an 'id'")
    if len(set(ids)) != len(ids):
        raise RuleCompileError("duplicate rule ids")
    for r in data["rules"]:
        if "expression" not in r:
            raise RuleCompileError(f"rule {r['id']!r} has no 'expression'")
        compile_expression(r["expression"])  # raises on invalid syntax

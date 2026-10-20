"""Rule set evaluation with deterministic ordering and cycle detection."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..detectors.base import context_snapshot
from ..findings import Category, Finding, Severity
from ..truth import TruthSeries
from .compiler import RULE_SCHEMA_VERSION, compile_expression, load_rule_file
from .nodes import EvalContext, Node, RuleError


class RuleCycleError(RuleError):
    pass


@dataclass
class Rule:
    """A named evidence rule producing :class:`~eii_cloud.findings.Finding` objects."""

    id: str
    expression: Node
    category: Category = Category.COMPOSITE
    severity: Severity = Severity.HIGH
    parameters: list[str] | None = None
    min_duration: int = 1
    description: str = ""
    version: str = "1"
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.category = Category(self.category)
        self.severity = Severity(self.severity)
        self.expression = compile_expression(self.expression)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Rule:
        d = dict(d)
        return cls(
            id=str(d.pop("id")), expression=d.pop("expression"), category=d.pop("category", "composite_evidence"),
            severity=d.pop("severity", "high"), parameters=d.pop("parameters", None),
            min_duration=int(d.pop("min_duration", 1)), description=d.pop("description", ""),
            version=str(d.pop("version", "1")), enabled=bool(d.pop("enabled", True)), tags=list(d.pop("tags", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "category": self.category.value, "severity": self.severity.value,
                "parameters": self.parameters, "min_duration": self.min_duration, "description": self.description,
                "version": self.version, "enabled": self.enabled, "tags": self.tags,
                "expression": self.expression.to_dict()}

    def dependencies(self) -> set[str]:
        return self.expression.rule_refs()

    def parameter_ids(self) -> list[str]:
        return list(self.parameters) if self.parameters else self.expression.parameters()


@dataclass
class RuleResult:
    rule: Rule
    truth: TruthSeries
    findings: list[Finding]
    lookahead: float


class RuleSet:
    """Ordered collection of rules (topologically sorted by ``rule`` references)."""

    def __init__(self, rules: Iterable[Rule] = (), schema_version: str = RULE_SCHEMA_VERSION,
                 name: str = "ruleset"):
        self.rules: list[Rule] = []
        self.schema_version = schema_version
        self.name = name
        for r in rules:
            self.add(r)

    def add(self, rule: Rule | Mapping[str, Any]) -> RuleSet:
        r = rule if isinstance(rule, Rule) else Rule.from_dict(rule)
        if any(x.id == r.id for x in self.rules):
            raise RuleError(f"duplicate rule id {r.id!r}")
        self.rules.append(r)
        return self

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def get(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise KeyError(rule_id)

    # ----------------------------------------------------------------- IO
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuleSet:
        return cls([Rule.from_dict(r) for r in data["rules"]], schema_version=str(data.get("schema_version", RULE_SCHEMA_VERSION)),
                   name=str(data.get("name", "ruleset")))

    @classmethod
    def from_yaml(cls, path: str | Path) -> RuleSet:
        return cls.from_dict(load_rule_file(path))

    @classmethod
    def from_yaml_text(cls, text: str) -> RuleSet:
        data = yaml.safe_load(text)
        from .compiler import validate_rule_document

        validate_rule_document(data)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "name": self.name, "rules": [r.to_dict() for r in self.rules]}

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    # -------------------------------------------------------------- ordering
    def ordered(self) -> list[Rule]:
        """Deterministic topological order; raises :class:`RuleCycleError` on cycles."""
        by_id = {r.id: r for r in self.rules}
        for r in self.rules:
            for dep in r.dependencies():
                if dep not in by_id:
                    raise RuleError(f"rule {r.id!r} references unknown rule {dep!r}")
        state: dict[str, int] = {}
        order: list[Rule] = []

        def visit(rid: str, stack: list[str]) -> None:
            st = state.get(rid, 0)
            if st == 2:
                return
            if st == 1:
                cycle = stack[stack.index(rid):] + [rid]
                raise RuleCycleError("rule dependency cycle: " + " -> ".join(cycle))
            state[rid] = 1
            for dep in sorted(by_id[rid].dependencies()):
                visit(dep, stack + [rid])
            state[rid] = 2
            order.append(by_id[rid])

        for r in self.rules:  # definition order first, then dependencies
            visit(r.id, [])
        return order

    # ------------------------------------------------------------ evaluation
    def evaluate(self, ctx: EvalContext) -> dict[str, RuleResult]:
        results: dict[str, RuleResult] = {}
        for rule in self.ordered():
            truth = rule.expression.evaluate(ctx)
            ctx.rule_results[rule.id] = truth
            if not rule.enabled:
                continue
            findings = _findings_for_rule(rule, truth, ctx)
            results[rule.id] = RuleResult(rule, truth, findings, rule.expression.lookahead(ctx))
        return results


def _findings_for_rule(rule: Rule, truth: TruthSeries, ctx: EvalContext) -> list[Finding]:
    ds = ctx.dataset
    out: list[Finding] = []
    params = rule.parameter_ids()
    for s, e in truth.runs():
        if e - s + 1 < rule.min_duration:
            continue
        evidence = {"expression": rule.expression.explain(ctx, s, e)}
        evidence["matched_event_ids"] = sorted(_collect(evidence["expression"], "matched_event_ids"))
        evidence["matched_finding_ids"] = sorted(_collect(evidence["expression"], "matched_finding_ids"))
        anchor_i = (s + e) // 2
        anchor_v = None
        if params and ds.has_parameter(params[0]):
            x = ds.value(params[0])[s:e + 1]
            if np.any(~np.isnan(x)):
                j = int(np.nanargmax(np.abs(x - np.nanmedian(x))))
                anchor_i, anchor_v = s + j, float(x[j])
        desc = rule.description or f"{rule.category.title}:\nRule '{rule.id}' is satisfied."
        out.append(Finding(
            asset_id=ds.asset_id, parameter_ids=params, category=rule.category,
            start_time=ds.time.label(s), end_time=ds.time.label(e), start_index=s, end_index=e,
            detector_id="rule_engine", rule_id=rule.id, severity=rule.severity, evidence=evidence,
            context_snapshot=context_snapshot(ds, ctx.context, s, e), reference_id=f"rule:{rule.id}@{rule.version}",
            description=desc, anchor_index=anchor_i, anchor_value=anchor_v,
            provenance={"rule": rule.to_dict(), "rule_version": rule.version},
        ))
    return out


def _collect(node: Mapping[str, Any], key: str) -> set[str]:
    found = set(node.get(key, []))
    for c in node.get("children", []):
        found |= _collect(c, key)
    return found

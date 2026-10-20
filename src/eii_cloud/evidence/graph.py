"""Evidence graph: how conditions, detector outputs and events support a finding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..findings import Finding


@dataclass
class EvidenceNode:
    id: str
    kind: str
    label: str
    status: str  # TRUE / FALSE / UNKNOWN / MIXED
    detail: dict[str, Any] = field(default_factory=dict)
    depth: int = 0


@dataclass
class EvidenceGraph:
    """Directed graph (parent -> child) of the evidence behind one finding."""

    finding_id: str
    nodes: list[EvidenceNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def from_finding(cls, finding: Finding) -> EvidenceGraph:
        g = cls(finding.finding_id)
        root_label = finding.rule_id or finding.detector_id
        g.nodes.append(EvidenceNode("root", "finding", f"{finding.category.title}\n{root_label}", "TRUE",
                                    {"start": finding.start_time, "end": finding.end_time}, 0))
        expr = finding.evidence.get("expression")
        if expr is not None:
            g._add(expr, "root", 1)
        else:
            for k, v in finding.evidence.items():
                nid = f"ev:{k}"
                g.nodes.append(EvidenceNode(nid, "evidence", f"{k}: {_short(v)}", "TRUE", {"value": v}, 1))
                g.edges.append(("root", nid))
        for eid in finding.evidence.get("matched_event_ids", []):
            nid = f"event:{eid}"
            if not any(n.id == nid for n in g.nodes):
                g.nodes.append(EvidenceNode(nid, "event", f"event {eid}", "TRUE", {}, max(n.depth for n in g.nodes) + 1))
                g.edges.append(("root", nid))
        return g

    def _add(self, node: dict[str, Any], parent: str, depth: int) -> None:
        nid = f"n{len(self.nodes)}"
        nt, nf, nu = node.get("n_true", 0), node.get("n_false", 0), node.get("n_unknown", 0)
        status = "TRUE" if nt and not nf and not nu else "FALSE" if nf and not nt else "UNKNOWN" if nu and not nt else "MIXED"
        self.nodes.append(EvidenceNode(nid, node.get("kind", "node"), _label(node), status,
                                       {k: v for k, v in node.items() if k not in ("children", "spec")}, depth))
        self.edges.append((parent, nid))
        for c in node.get("children", []):
            self._add(c, nid, depth + 1)

    def to_dict(self) -> dict[str, Any]:
        return {"finding_id": self.finding_id,
                "nodes": [{"id": n.id, "kind": n.kind, "label": n.label, "status": n.status, "depth": n.depth,
                           "detail": n.detail} for n in self.nodes],
                "edges": [{"from": a, "to": b} for a, b in self.edges]}


def _short(v: Any, n: int = 24) -> str:
    s = str(v)
    return s if len(s) <= n else s[:n - 1] + "…"


def _label(node: dict[str, Any]) -> str:
    kind = node.get("kind", "node")
    spec = node.get("spec", {})
    if kind == "signal":
        return f"{spec['signal']} {spec['op']} {spec['value']}"
    if kind == "outside_envelope":
        return f"outside envelope\n{spec['outside_envelope']['parameter']}"
    if kind == "event":
        ids = node.get("matched_event_ids", [])
        return f"event {spec['event']['event_type']}" + (f"\n{', '.join(ids[:3])}" if ids else "")
    if kind in ("before", "after", "within", "persists_for", "count_within"):
        w = spec[kind].get("window")
        return f"{kind.upper()} (w={w})"
    if kind == "missing_when_expected":
        return f"missing when expected\n{spec['missing_when_expected']}"
    if kind == "contradicts_context":
        return f"contradicts context\n{spec['contradicts_context']['parameter']}"
    if kind == "finding":
        f = spec["finding"]
        return "finding " + (f.get("detector") or f.get("category") or "")
    if kind == "rule":
        return f"rule {spec['rule']}"
    if kind == "context":
        return f"regime {spec['context']}"
    return kind.upper()

"""Reproducible experiment runs from a serialisable configuration."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..analysis import EIIEngine
from ..data import EIIDataset
from ..evidence.provenance import _json_default, config_hash, provenance_record
from ..export import ExportManager
from ..findings import AnalysisResult, findings_equal
from ..rules import RuleSet
from ..visualization.annotations import annotations_document
from ..visualization.eii_chart import EIICloudChart
from . import synthetic


class ExperimentRunner:
    """Run an analysis described by a config mapping (YAML-loadable) and write all artefacts.

    Config keys::

        name: my_run
        dataset: {synthetic: {scenario: reference, seed: 0}}     # or {path: saved_dataset_dir}
        reference: {path: ...}                                   # optional
        context: {...}                                           # ContextDefinition.from_dict spec, optional
        envelopes: {name: {method: ..., parameter: ...}}
        detectors: [{type: threshold, parameter: gpu_temp, envelope: name}, ...]
        rules: path/to/rules.yaml  |  {rules: [...]}
        charts: [{parameter: gpu_temp, envelope: name, modes: [annotated, clean, flags_only]}]
        export: {formats: [svg, pdf, tiff, png], dpi: 300}

    A synthetic ``reference`` scenario fills in context/envelopes/rules when they are omitted.
    """

    def __init__(self, config: Mapping[str, Any], out_dir: str | Path):
        self.config = json.loads(json.dumps(dict(config), default=_json_default))
        self.out_dir = Path(out_dir)

    @classmethod
    def from_yaml(cls, path: str | Path, out_dir: str | Path | None = None) -> ExperimentRunner:
        cfg = yaml.safe_load(Path(path).read_text())
        out = out_dir or cfg.get("out_dir") or Path(path).with_suffix("")
        return cls(cfg, out)

    # ------------------------------------------------------------ assembly
    def load_dataset(self, spec: Mapping[str, Any]) -> tuple[EIIDataset, Any]:
        if "path" in spec:
            return EIIDataset.load(spec["path"]), None
        if "synthetic" in spec:
            s = spec["synthetic"]
            if s.get("scenario", "reference") != "reference":
                raise ValueError("only the 'reference' synthetic scenario is built in")
            sc = synthetic.make_reference_scenario(seed=int(s.get("seed", 0)), n=int(s.get("n", 100)),
                                                   noise=float(s.get("noise", 0.9)))
            ds = sc.dataset
            for inj in s.get("inject", []):
                ds = synthetic.inject(ds, **inj)
            return ds, sc
        raise ValueError("dataset spec needs 'path' or 'synthetic'")

    def build(self) -> tuple[EIIEngine, EIIDataset, EIIDataset | None, Any]:
        cfg = self.config
        dataset, scenario = self.load_dataset(cfg["dataset"])
        reference = None
        if "reference" in cfg:
            reference, _ = self.load_dataset(cfg["reference"])
        elif scenario is not None:
            reference = scenario.reference
        context = cfg.get("context")
        if context is None and scenario is not None:
            context = scenario.context
        envelopes = cfg.get("envelopes")
        if envelopes is None and scenario is not None:
            envelopes = scenario.envelopes
        rules_spec = cfg.get("rules")
        if isinstance(rules_spec, str):
            rules = RuleSet.from_yaml(rules_spec)
        elif isinstance(rules_spec, Mapping):
            rules = RuleSet.from_dict(rules_spec)
        elif scenario is not None:
            rules = scenario.rules
        else:
            rules = RuleSet()
        detectors = cfg.get("detectors")
        if detectors is None and scenario is not None:
            detectors = default_detectors(list(envelopes or {}))
        engine = EIIEngine(envelopes=envelopes or {}, detectors=detectors or [], rules=rules, context=context,
                           event_tolerance=cfg.get("event_tolerance"), name=cfg.get("name", "experiment"))
        return engine, dataset, reference, scenario

    # ---------------------------------------------------------------- run
    def run(self, write: bool = True) -> AnalysisResult:
        engine, dataset, reference, scenario = self.build()
        result = engine.run(dataset, reference)
        if not write:
            return result
        self.out_dir.mkdir(parents=True, exist_ok=True)
        dataset.save(self.out_dir / "dataset")
        if reference is not None:
            reference.save(self.out_dir / "reference")
        result.to_json(self.out_dir / "result.json")
        (self.out_dir / "rules.yaml").write_text(engine.rules.to_yaml())
        exp_cfg = self.config.get("export", {})
        em = ExportManager(self.out_dir / "charts", dpi=int(exp_cfg.get("dpi", 300)))
        chart_files: dict[str, Any] = {}
        import matplotlib.pyplot as plt

        for ch in self.config.get("charts") or default_charts(dataset, engine):
            p = ch["parameter"]
            doc = annotations_document(result, dataset, p)
            em.export_json(doc, f"{p}_annotations")
            for mode in ch.get("modes", ["annotated", "clean", "flags_only"]):
                chart = EIICloudChart(dataset, result, p, envelope=ch.get("envelope"), mode=mode,
                                      annotation_panel=bool(ch.get("annotation_panel", False)))
                fig = chart.render()
                files = em.export(fig, f"{p}_{mode}", formats=exp_cfg.get("formats", ["svg", "pdf", "tiff", "png"]),
                                  metadata={"analysis_id": result.analysis_id, "parameter": p, "mode": mode,
                                            "annotation_ids": chart.annotation_ids()})
                chart_files[f"{p}_{mode}"] = {k: str(v) for k, v in files.items()}
                plt.close(fig)
        manifest = {
            "name": self.config.get("name", "experiment"),
            "config": self.config,
            "config_sha256": config_hash(self.config),
            "engine_config_sha256": config_hash(engine.config()),
            "analysis_id": result.analysis_id,
            "dataset_manifest": dataset.manifest(),
            "reference_manifest": reference.manifest() if reference is not None else None,
            "n_findings": len(result.findings),
            "finding_ids": sorted(f.finding_id for f in result.findings),
            "charts": chart_files,
            "ground_truth": scenario.ground_truth if scenario is not None else None,
            "provenance": provenance_record(),
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
        return result

    # ----------------------------------------------------------- reproduce
    @classmethod
    def reproduce(cls, run_dir: str | Path) -> dict[str, Any]:
        """Re-run a saved experiment from its manifest and compare the findings."""
        run_dir = Path(run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        previous = AnalysisResult.from_json(run_dir / "result.json")
        runner = cls(manifest["config"], run_dir / "_reproduction")
        engine, dataset, reference, _ = runner.build()
        saved = EIIDataset.load(run_dir / "dataset")
        if saved.content_hash() != dataset.content_hash():
            raise ValueError("regenerated dataset differs from the saved dataset manifest")
        result = engine.run(dataset, reference)
        same = findings_equal(result.findings, previous.findings)
        return {"identical": same, "analysis_id_previous": previous.analysis_id, "analysis_id_now": result.analysis_id,
                "n_previous": len(previous.findings), "n_now": len(result.findings),
                "config_sha256_match": config_hash(runner.config) == manifest["config_sha256"]}


def default_detectors(envelope_names: list[str]) -> list[dict[str, Any]]:
    dets: list[dict[str, Any]] = [
        {"type": "missingness", "parameter": "gpu_temp", "detector_id": "missing:gpu_temp"},
        {"type": "flatline", "parameter": "gpu_temp", "min_run": 8, "tolerance": 0.0, "detector_id": "flatline:gpu_temp"},
        {"type": "context_contradiction", "parameter": "gpu_temp", "detector_id": "context:gpu_temp"},
    ]
    if "gpu_temp_static" in envelope_names:
        dets.insert(0, {"type": "threshold", "parameter": "gpu_temp", "envelope": "gpu_temp_static",
                        "detector_id": "threshold:gpu_temp"})
    return dets


def default_charts(dataset: EIIDataset, engine: EIIEngine) -> list[dict[str, Any]]:
    charts = []
    for name, spec in engine.envelope_specs.items():
        p = spec.parameter if hasattr(spec, "parameter") else spec["parameter"]
        charts.append({"parameter": p, "envelope": name, "modes": ["annotated", "clean", "flags_only"]})
        break
    return charts or [{"parameter": dataset.parameters[0]}]

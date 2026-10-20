"""Command-line interface: ``eii-cloud demo | run | reproduce | validate-rules | protocols | list``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._version import __version__


def _demo(args: argparse.Namespace) -> int:
    from .experiments.runner import ExperimentRunner

    cfg = {"name": "eii_cloud_demo", "dataset": {"synthetic": {"scenario": "reference", "seed": args.seed}},
           "charts": [{"parameter": "gpu_temp", "envelope": "gpu_temp_static",
                       "modes": ["annotated", "clean", "flags_only"], "annotation_panel": args.panel}],
           "export": {"formats": args.formats.split(","), "dpi": args.dpi}}
    result = ExperimentRunner(cfg, args.out).run()
    print(result.summary())
    print(f"\nArtefacts written to {Path(args.out).resolve()}")
    return 0


def _run(args: argparse.Namespace) -> int:
    from .experiments.runner import ExperimentRunner

    runner = ExperimentRunner.from_yaml(args.config, args.out)
    result = runner.run()
    print(result.summary())
    print(f"\nArtefacts written to {runner.out_dir.resolve()}")
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    from .experiments.runner import ExperimentRunner

    rep = ExperimentRunner.reproduce(args.run_dir)
    print(json.dumps(rep, indent=2))
    return 0 if rep["identical"] else 1


def _validate(args: argparse.Namespace) -> int:
    from .rules import RuleSet, load_rule_file

    data = load_rule_file(args.rules)
    rs = RuleSet.from_dict(data)
    order = [r.id for r in rs.ordered()]
    print(f"{len(rs)} rule(s) valid; evaluation order: {order}")
    return 0


def _protocols(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from .experiments import protocols

    fn = {"all": protocols.run_all, "exp00": protocols.exp00_reference_reproduction,
          "exp01": protocols.exp01_context_vs_static, "exp03": protocols.exp03_observability_degradation,
          "exp05": protocols.exp05_export_reproducibility, "exp06": protocols.exp06_batch_vs_streaming}[args.which]
    rep = fn(seed=args.seed)
    print(json.dumps(rep, indent=2, default=str))
    return 0


def _list(args: argparse.Namespace) -> int:
    from .detectors import list_detectors
    from .envelopes import ENVELOPE_METHODS
    from .visualization import list_charts

    print("detectors:", ", ".join(list_detectors()))
    print("envelope methods:", ", ".join(sorted(ENVELOPE_METHODS)))
    print("charts:", ", ".join(list_charts()))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eii-cloud", description="Early Instability Indicator Cloud toolkit")
    p.add_argument("--version", action="version", version=f"eii-cloud {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the reference synthetic scenario and export all artefacts")
    d.add_argument("--out", default="eii_cloud_demo")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--formats", default="svg,pdf,tiff,png")
    d.add_argument("--dpi", type=int, default=300)
    d.add_argument("--panel", action="store_true", help="put annotation text in a side panel")
    d.set_defaults(fn=_demo)

    r = sub.add_parser("run", help="run an experiment from a YAML config")
    r.add_argument("config")
    r.add_argument("--out", default=None)
    r.set_defaults(fn=_run)

    rp = sub.add_parser("reproduce", help="re-run a saved experiment and compare findings")
    rp.add_argument("run_dir")
    rp.set_defaults(fn=_reproduce)

    v = sub.add_parser("validate-rules", help="validate a YAML rule file (syntax, ids, cycles)")
    v.add_argument("rules")
    v.set_defaults(fn=_validate)

    pr = sub.add_parser("protocols", help="run built-in validation protocols on synthetic data")
    pr.add_argument("which", choices=["all", "exp00", "exp01", "exp03", "exp05", "exp06"], nargs="?", default="all")
    pr.add_argument("--seed", type=int, default=0)
    pr.set_defaults(fn=_protocols)

    ls = sub.add_parser("list", help="list registered detectors, envelope methods and charts")
    ls.set_defaults(fn=_list)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

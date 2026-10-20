# Specification → implementation map (v0.3 spec, package 0.1.0)

| Spec section | Where |
|---|---|
| 2 Core model, `Finding` fields | `eii_cloud/findings.py` |
| 3 Envelopes, 9 methods, support/coverage | `eii_cloud/envelopes.py` |
| 3.2 fitting on reference periods, freezing | `reference=` argument of `EIIEngine.run`, `ExperimentRunner` saves the reference |
| 4 Categories A1–A7 | `Category` enum; detectors in `eii_cloud/detectors/`; A7 via `eii_cloud/rules/` |
| 5.1 evidence graph, operators | `rules/nodes.py`, `evidence/graph.py` |
| 5.2 example composite rule | `experiments/synthetic.py::_rules`, `examples/rules/gpu_instability.yaml` |
| 5.3 three-valued logic | `truth.py` |
| 5.4 streaming, watermark, late events, cycle detection | `analysis.py::StreamingEngine`, `rules/engine.py::RuleSet.ordered` |
| 6 Results & provenance | `findings.py::AnalysisResult`, `evidence/provenance.py` |
| 7 Visualization engine, modes, layers | `visualization/eii_chart.py`, `visualization/annotations.py` |
| 7.3 exports SVG/PDF/TIFF/JSON | `export.py` |
| 8.1 heuristic detectors | `detectors/statistical.py`, `detectors/temporal.py`, `detectors/contextual.py` |
| 8.2 plug-ins | `register_detector`, `Forecaster` protocol, `regression_residual(model=...)` |
| 8.3 chart registry | `visualization/registry.py` |
| 10 EXP-00/01/03/05/06 | `experiments/protocols.py` (`eii-cloud protocols`) |
| 11 metrics | `evaluation.py` |
| 12 quality gates | `tests/`, `.github/workflows/ci.yml` |
| 14 definition of completion | `examples/reference_chart.py`, `eii-cloud demo` / `reproduce` |

Not yet implemented (planned for 0.2/0.3): matrix profile, STL, isolation forest, PCA
residuals, HMM, Bayesian change point, survival models, graph-based detectors,
foundation-model plug-ins, graphical rule designer, EXP-02/04/07 drivers on real data.

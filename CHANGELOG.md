# Changelog

## 0.1.0 (2026-10-20)

First experimental release (specification v0.3, release-plan stages v0.1 + v0.2 core):

- `EIIDataset` with acquisition schedule, context, event log with collector coverage, content hashes.
- Nine envelope construction methods with support/coverage metadata.
- Ten heuristic detectors (A1–A6) plus registry.
- Three-valued rule engine with temporal operators, YAML compiler, cycle detection, evidence provenance.
- `Finding` / `AnalysisResult` with stable identifiers and JSON schema 1.0.
- `EIICloudChart` (annotated / clean / flags_only / annotations_json) and nine auxiliary charts.
- SVG / PDF / TIFF / PNG export with shared annotation ids and JSON sidecars.
- Evaluation metrics, synthetic reference scenario, fault injection, experiment runner with reproduce.
- Streaming replay with watermark finalisation (EXP-06).

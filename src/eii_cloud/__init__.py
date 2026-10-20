"""EII_cloud: Early Instability Indicator Cloud.

Context-aware, explainable instability detection, evidence fusion and scientific
visualization for predictive-maintenance research.  See the README for a walkthrough.
"""
from ._version import SCHEMA_VERSION, __version__
from .analysis import EIIEngine, StreamingEngine, split_chunks
from .context import ContextDefinition, Expectation, RegimeRule
from .data import EIIDataset
from .detectors import (
                        BaseDetector,
                        ChangePointDetector,
                        ContextContradictionDetector,
                        CrossSignalConstraintDetector,
                        CUSUMDetector,
                        EWMADetector,
                        FlatlineDetector,
                        MissingnessDetector,
                        RateOfChangeDetector,
                        RobustDeviationDetector,
                        ThresholdDetector,
                        build_detector,
                        get_detector,
                        list_detectors,
                        register_detector,
)
from .envelopes import (
                        ENVELOPE_METHODS,
                        SeasonalNaiveForecaster,
                        ValidityEnvelope,
                        build_envelope,
                        contextual_quantiles,
                        external_envelope,
                        fixed_bounds,
                        forecast_interval,
                        physical_constraints,
                        regime_lookup,
                        regression_residual,
                        rolling_robust,
                        scheduled_bounds,
)
from .evidence import EventLog, EvidenceGraph
from .export import ExportManager
from .findings import AnalysisResult, Category, Finding, Severity, findings_equal
from .rules import C, Rule, RuleSet
from .truth import FALSE, TRUE, UNKNOWN, TruthSeries, TruthStatus
from .visualization import (
                        AnomalyScoreChart,
                        ChangePointChart,
                        ChartStyle,
                        ContextComparisonChart,
                        ControlChart,
                        DetectorComparisonChart,
                        EIICloudChart,
                        EventSequenceChart,
                        EvidenceGraphChart,
                        Layers,
                        MissingnessTimeline,
                        MultiSignalChart,
                        annotations_document,
                        build_annotations,
                        write_annotations_json,
)

__all__ = [
    "__version__", "SCHEMA_VERSION",
    "EIIDataset", "EventLog", "ContextDefinition", "Expectation", "RegimeRule",
    "ValidityEnvelope", "ENVELOPE_METHODS", "build_envelope", "fixed_bounds", "scheduled_bounds", "rolling_robust",
    "contextual_quantiles", "regime_lookup", "regression_residual", "physical_constraints", "external_envelope",
    "forecast_interval", "SeasonalNaiveForecaster",
    "BaseDetector", "register_detector", "get_detector", "list_detectors", "build_detector",
    "ThresholdDetector", "RobustDeviationDetector", "EWMADetector", "CUSUMDetector", "FlatlineDetector",
    "MissingnessDetector", "RateOfChangeDetector", "ChangePointDetector", "ContextContradictionDetector",
    "CrossSignalConstraintDetector",
    "C", "Rule", "RuleSet", "TruthSeries", "TruthStatus", "TRUE", "FALSE", "UNKNOWN",
    "EIIEngine", "StreamingEngine", "split_chunks", "AnalysisResult", "Finding", "Category", "Severity", "findings_equal",
    "EvidenceGraph", "ExportManager",
    "EIICloudChart", "Layers", "ChartStyle", "AnomalyScoreChart", "ControlChart", "ChangePointChart",
    "MissingnessTimeline", "EventSequenceChart", "MultiSignalChart", "ContextComparisonChart",
    "DetectorComparisonChart", "EvidenceGraphChart", "annotations_document", "build_annotations",
    "write_annotations_json",
]

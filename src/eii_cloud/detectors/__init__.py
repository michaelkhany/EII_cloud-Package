"""Detector registry and built-in heuristic detectors."""
from .base import (
    BaseDetector,
    DetectorOutput,
    build_detector,
    get_detector,
    list_detectors,
    register_detector,
)
from .contextual import ContextContradictionDetector, CrossSignalConstraintDetector
from .statistical import CUSUMDetector, EWMADetector, RobustDeviationDetector, ThresholdDetector
from .temporal import ChangePointDetector, FlatlineDetector, MissingnessDetector, RateOfChangeDetector

__all__ = [
    "BaseDetector", "DetectorOutput", "build_detector", "get_detector", "list_detectors", "register_detector",
    "ThresholdDetector", "RobustDeviationDetector", "EWMADetector", "CUSUMDetector",
    "FlatlineDetector", "MissingnessDetector", "RateOfChangeDetector", "ChangePointDetector",
    "ContextContradictionDetector", "CrossSignalConstraintDetector",
]

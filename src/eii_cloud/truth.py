"""Three-valued (Kleene) truth series.

Every condition, operator and detector in ``eii_cloud`` evaluates to a
:class:`TruthSeries` aligned with the dataset time axis.  Each sample is one of

* ``TRUE`` (1): sufficient evidence satisfies the condition,
* ``FALSE`` (0): sufficient *valid* evidence contradicts the condition,
* ``UNKNOWN`` (-1): required evidence is unavailable or insufficient.

Negation never turns UNKNOWN into TRUE or FALSE, so an unobserved event is never
silently interpreted as an absent event.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from enum import Enum

import numpy as np

TRUE: int = 1
FALSE: int = 0
UNKNOWN: int = -1


class TruthStatus(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_int(cls, v: int) -> TruthStatus:
        return {1: cls.TRUE, 0: cls.FALSE, -1: cls.UNKNOWN}[int(v)]


class TruthSeries:
    """Vector of three-valued truth values stored as ``int8``."""

    __slots__ = ("values",)

    def __init__(self, values: Iterable[int] | np.ndarray):
        arr = np.asarray(values, dtype=np.int8)
        if arr.ndim != 1:
            raise ValueError("TruthSeries must be one-dimensional")
        if not np.all(np.isin(arr, (1, 0, -1))):
            raise ValueError("TruthSeries values must be 1 (TRUE), 0 (FALSE) or -1 (UNKNOWN)")
        self.values = arr

    # --------------------------------------------------------- constructors
    @classmethod
    def from_bool(cls, mask: np.ndarray, unknown: np.ndarray | None = None) -> TruthSeries:
        """Build from a boolean mask; samples where ``unknown`` is True become UNKNOWN."""
        mask = np.asarray(mask, dtype=bool)
        out = mask.astype(np.int8)
        if unknown is not None:
            out = np.where(np.asarray(unknown, dtype=bool), np.int8(-1), out).astype(np.int8)
        return cls(out)

    @classmethod
    def constant(cls, n: int, value: int) -> TruthSeries:
        return cls(np.full(n, value, dtype=np.int8))

    # ----------------------------------------------------------- Kleene ops
    def __and__(self, other: TruthSeries) -> TruthSeries:
        a, b = self.values, other.values
        self._check(other)
        out = np.where((a == 0) | (b == 0), 0, np.where((a == 1) & (b == 1), 1, -1))
        return TruthSeries(out)

    def __or__(self, other: TruthSeries) -> TruthSeries:
        a, b = self.values, other.values
        self._check(other)
        out = np.where((a == 1) | (b == 1), 1, np.where((a == 0) & (b == 0), 0, -1))
        return TruthSeries(out)

    def __invert__(self) -> TruthSeries:
        a = self.values
        out = np.where(a == 1, 0, np.where(a == 0, 1, -1))
        return TruthSeries(out)

    def _check(self, other: TruthSeries) -> None:
        if len(other.values) != len(self.values):
            raise ValueError("TruthSeries length mismatch")

    # ------------------------------------------------------------ queries
    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[int]:
        return iter(self.values.tolist())

    def __getitem__(self, i: int) -> int:
        return int(self.values[i])

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        return isinstance(other, TruthSeries) and np.array_equal(self.values, other.values)

    @property
    def is_true(self) -> np.ndarray:
        return self.values == 1

    @property
    def is_false(self) -> np.ndarray:
        return self.values == 0

    @property
    def is_unknown(self) -> np.ndarray:
        return self.values == -1

    def fraction_unknown(self) -> float:
        return float(np.mean(self.is_unknown)) if len(self.values) else 0.0

    def runs(self, value: int = TRUE) -> list[tuple[int, int]]:
        """Inclusive ``(start_index, end_index)`` pairs of maximal runs equal to ``value``."""
        return runs_of(self.values == value)

    def status_at(self, i: int) -> TruthStatus:
        return TruthStatus.from_int(self.values[i])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = len(self.values)
        return f"TruthSeries(n={n}, true={int(self.is_true.sum())}, false={int(self.is_false.sum())}, unknown={int(self.is_unknown.sum())})"


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index pairs for maximal runs of True in a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def kleene_all(series: list[TruthSeries]) -> TruthSeries:
    if not series:
        raise ValueError("kleene_all requires at least one series")
    out = series[0]
    for s in series[1:]:
        out = out & s
    return out


def kleene_any(series: list[TruthSeries]) -> TruthSeries:
    if not series:
        raise ValueError("kleene_any requires at least one series")
    out = series[0]
    for s in series[1:]:
        out = out | s
    return out

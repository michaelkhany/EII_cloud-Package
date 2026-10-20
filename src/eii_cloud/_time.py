"""Time-axis helper.

All temporal reasoning inside ``eii_cloud`` is performed on a float64 array ``t``
(seconds since the Unix epoch for datetime indices, or the raw numeric value for
numeric indices). The original index is kept for labelling and plotting so that
JSON output reproduces the user's own timestamps.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

OffsetLike = int | float | str | pd.Timedelta | np.timedelta64


def _epoch_seconds(idx: pd.DatetimeIndex) -> np.ndarray:
    """Seconds since the Unix epoch regardless of the index resolution (s/ms/us/ns)."""
    return idx.as_unit("ns").asi8.astype(np.float64) / 1e9


class TimeAxis:
    """Strictly increasing event-time axis.

    Parameters
    ----------
    index:
        Any sequence convertible to a :class:`pandas.Index`. Datetime-like
        values become a :class:`pandas.DatetimeIndex`; numeric values are used
        as-is (e.g. "time window" counters as in the reference chart).
    """

    __slots__ = ("index", "t", "kind")

    def __init__(self, index: Any):
        idx = pd.Index(index)
        if isinstance(idx, pd.DatetimeIndex):
            self.kind = "datetime"
            self.t = _epoch_seconds(idx)
        elif pd.api.types.is_numeric_dtype(idx.dtype) and not pd.api.types.is_bool_dtype(idx.dtype):
            self.kind = "numeric"
            self.t = np.asarray(idx, dtype=np.float64)
        else:
            try:
                idx = pd.DatetimeIndex(pd.to_datetime(idx))
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise TypeError(f"Unsupported time index dtype {idx.dtype!r}") from exc
            self.kind = "datetime"
            self.t = _epoch_seconds(idx)
        if len(idx) == 0:
            raise ValueError("Time axis must contain at least one sample")
        if np.any(np.isnan(self.t)):
            raise ValueError("Time axis contains NaT/NaN values")
        if len(idx) > 1 and not np.all(np.diff(self.t) > 0):
            raise ValueError("Time axis must be strictly increasing (sorted, no duplicates)")
        self.index = idx

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.t)

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def is_datetime(self) -> bool:
        return self.kind == "datetime"

    @property
    def median_step(self) -> float:
        if len(self.t) < 2:
            return float("nan")
        return float(np.median(np.diff(self.t)))

    # ----------------------------------------------------------------- offsets
    def offset(self, value: OffsetLike) -> float:
        """Convert a user offset (number, ``"5min"``, Timedelta) to axis seconds/units."""
        if isinstance(value, (pd.Timedelta, np.timedelta64)):
            if not self.is_datetime:
                raise TypeError("Timedelta offsets require a datetime time axis")
            return float(pd.Timedelta(value).total_seconds())
        if isinstance(value, str):
            if not self.is_datetime:
                try:
                    return float(value)
                except ValueError as exc:
                    raise TypeError("String offsets require a datetime time axis") from exc
            return float(pd.Timedelta(value).total_seconds())
        if isinstance(value, bool):
            raise TypeError("Boolean is not a valid offset")
        return float(value)

    # ------------------------------------------------------------------ labels
    def label(self, i: int) -> Any:
        """JSON-serialisable label of sample ``i``."""
        v = self.index[int(i)]
        if self.is_datetime:
            return pd.Timestamp(v).isoformat()
        return float(v) if isinstance(v, (np.floating, float)) else int(v)

    def labels(self, indices: Any) -> list:
        return [self.label(i) for i in np.asarray(indices, dtype=int)]

    def from_label(self, label: Any) -> float:
        """Convert an external label (ISO string or number) to float axis units."""
        if self.is_datetime:
            ts = pd.Timestamp(label)
            if ts.tzinfo is None and self.index.tz is not None:
                ts = ts.tz_localize(self.index.tz)
            return float(ts.as_unit("ns").value) / 1e9
        return float(label)

    def plot_values(self) -> Any:
        """Values usable on a Matplotlib axis."""
        return self.index.to_numpy()

    def locate(self, label: Any, side: str = "left") -> int:
        return int(np.searchsorted(self.t, self.from_label(label), side=side))

    def slice_indices(self, start: Any | None, end: Any | None) -> tuple[int, int]:
        lo = 0 if start is None else self.locate(start, "left")
        hi = self.n if end is None else self.locate(end, "right")
        return lo, hi

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TimeAxis(kind={self.kind!r}, n={self.n}, start={self.label(0)!r}, end={self.label(self.n - 1)!r})"

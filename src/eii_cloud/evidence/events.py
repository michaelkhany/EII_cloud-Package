"""Discrete evidence: log events, maintenance records, external observations."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

REQUIRED = ("event_id", "event_type", "timestamp")


class EventLog:
    """Table of discrete events plus optional *collector coverage* intervals.

    Coverage intervals state when an event source was actually able to report.
    Outside coverage, ``event_present`` evaluates to UNKNOWN instead of FALSE, which is
    what prevents "no error logged" from being treated as evidence when the log
    collector was down.
    """

    def __init__(self, frame: pd.DataFrame | None = None, coverage: pd.DataFrame | None = None):
        if frame is None or len(frame) == 0:
            frame = pd.DataFrame({"event_id": pd.Series([], dtype=str), "event_type": pd.Series([], dtype=str),
                                  "timestamp": pd.Series([], dtype=object), "asset_id": pd.Series([], dtype=str),
                                  "source": pd.Series([], dtype=str)})
        frame = frame.copy().reset_index(drop=True)
        for col in REQUIRED:
            if col not in frame.columns:
                raise ValueError(f"EventLog requires column {col!r}")
        if "asset_id" not in frame.columns:
            frame["asset_id"] = ""
        if "source" not in frame.columns:
            frame["source"] = frame["event_type"].astype(str)
        if frame["event_id"].duplicated().any():
            dups = frame.loc[frame["event_id"].duplicated(), "event_id"].tolist()
            raise ValueError(f"duplicate event_id values: {dups[:5]}")
        self.frame = frame
        if coverage is None:
            coverage = pd.DataFrame({"source": pd.Series([], dtype=str), "start": pd.Series([], dtype=object),
                                     "end": pd.Series([], dtype=object)})
        self.coverage = coverage.reset_index(drop=True)

    # ------------------------------------------------------------ builders
    @classmethod
    def empty(cls) -> EventLog:
        return cls()

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]], coverage: Iterable[Mapping[str, Any]] | None = None) -> EventLog:
        recs = list(records)
        if recs and not all("event_id" in r for r in recs):
            recs = [dict(r, event_id=r.get("event_id", f"ev{i:06d}")) for i, r in enumerate(recs)]
        frame = pd.DataFrame(recs) if recs else None
        cov = pd.DataFrame(list(coverage)) if coverage else None
        return cls(frame, cov)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, coverage: pd.DataFrame | None = None) -> EventLog:
        if "timestamp" in frame.columns and frame["timestamp"].dtype == object:
            try:
                frame = frame.assign(timestamp=pd.to_numeric(frame["timestamp"]))
            except (TypeError, ValueError):
                frame = frame.assign(timestamp=pd.to_datetime(frame["timestamp"]))
        return cls(frame, coverage)

    def with_coverage(self, source: str, start: Any, end: Any) -> EventLog:
        cov = pd.concat([self.coverage, pd.DataFrame([{"source": source, "start": start, "end": end}])], ignore_index=True)
        return EventLog(self.frame, cov)

    # -------------------------------------------------------------- queries
    def __len__(self) -> int:
        return len(self.frame)

    def types(self) -> list[str]:
        return sorted(self.frame["event_type"].astype(str).unique().tolist())

    def select(self, event_type: str | None = None, asset_id: str | None = None, source: str | None = None) -> pd.DataFrame:
        f = self.frame
        if event_type is not None:
            f = f[f["event_type"].astype(str) == event_type]
        if asset_id is not None and "asset_id" in f.columns:
            f = f[(f["asset_id"].astype(str) == asset_id) | (f["asset_id"].astype(str) == "")]
        if source is not None:
            f = f[f["source"].astype(str) == source]
        return f

    def deduplicate(self, tolerance: float | pd.Timedelta = 0, keys: tuple[str, ...] = ("event_type", "asset_id")) -> EventLog:
        """Drop events of equal ``keys`` whose timestamps lie within ``tolerance`` of a kept one."""
        f = self.frame.sort_values("timestamp").reset_index(drop=True)
        keep = np.ones(len(f), dtype=bool)
        last: dict[tuple, Any] = {}
        for i, row in f.iterrows():
            key = tuple(str(row[k]) for k in keys if k in f.columns)
            ts = row["timestamp"]
            prev = last.get(key)
            if prev is not None and _within(ts, prev, tolerance):
                keep[i] = False
            else:
                last[key] = ts
        return EventLog(f[keep].reset_index(drop=True), self.coverage)

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()

    def content_hash(self) -> str:
        h = hashlib.sha256()
        if len(self.frame):
            h.update(self.frame.astype(str).to_csv(index=False).encode())
        if len(self.coverage):
            h.update(self.coverage.astype(str).to_csv(index=False).encode())
        return h.hexdigest()

    def __repr__(self) -> str:  # pragma: no cover
        return f"EventLog(n={len(self)}, types={self.types()})"


def _within(a: Any, b: Any, tol: Any) -> bool:
    try:
        return abs(a - b) <= tol
    except TypeError:
        return abs(pd.Timestamp(a) - pd.Timestamp(b)) <= pd.Timedelta(tol)

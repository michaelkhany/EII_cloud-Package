"""Dataset container: telemetry, operating context, observation availability and quality."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._time import TimeAxis
from ._version import SCHEMA_VERSION, __version__
from .evidence.events import EventLog


class EIIDataset:
    """Telemetry for one asset on a shared event-time axis.

    Parameters
    ----------
    time:
        Time index (datetime-like or numeric), strictly increasing.
    values:
        ``DataFrame`` of parameter measurements (NaN = not observed).
    context:
        Optional ``DataFrame`` of operating-context columns (categorical or numeric).
    expected:
        Optional boolean ``DataFrame`` (same columns as ``values``) telling whether the
        acquisition schedule *expects* an observation at each sample. Defaults to all True.
        Set it to False for scheduled shutdowns or intentionally disabled sensors so that
        a gap there is **not** reported as "missing when expected".
    quality:
        Optional ``DataFrame`` of per-parameter quality flags/scores (user-defined).
    units:
        Mapping parameter -> unit string for axis labels.
    events:
        Optional :class:`~eii_cloud.evidence.events.EventLog` with logs/maintenance events.
    """

    def __init__(
        self,
        time: Any,
        values: pd.DataFrame | Mapping[str, Sequence[float]],
        context: pd.DataFrame | Mapping[str, Sequence[Any]] | None = None,
        expected: pd.DataFrame | Mapping[str, Sequence[bool]] | None = None,
        quality: pd.DataFrame | None = None,
        units: Mapping[str, str] | None = None,
        events: EventLog | None = None,
        asset_id: str = "asset",
        name: str = "dataset",
        attrs: Mapping[str, Any] | None = None,
    ):
        self.time = TimeAxis(time)
        n = self.time.n
        vals = pd.DataFrame(values)
        if len(vals) != n:
            raise ValueError(f"values has {len(vals)} rows but time has {n}")
        self.values = vals.reset_index(drop=True).astype("float64")
        if self.values.columns.duplicated().any():
            raise ValueError("duplicate parameter names")
        self.context = (
            pd.DataFrame(context).reset_index(drop=True)
            if context is not None
            else pd.DataFrame(index=range(n))
        )
        if len(self.context) != n:
            raise ValueError("context length mismatch")
        if expected is None:
            self.expected = pd.DataFrame(True, index=range(n), columns=self.values.columns)
        else:
            exp = pd.DataFrame(expected).reset_index(drop=True)
            for col in self.values.columns:
                if col not in exp.columns:
                    exp[col] = True
            self.expected = exp[self.values.columns].astype(bool)
            if len(self.expected) != n:
                raise ValueError("expected length mismatch")
        self.quality = quality.reset_index(drop=True) if quality is not None else None
        self.units = dict(units or {})
        self.events = events if events is not None else EventLog.empty()
        self.asset_id = asset_id
        self.name = name
        self.attrs = dict(attrs or {})

    # ------------------------------------------------------------ construction
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        time_column: str | None = None,
        parameter_columns: Iterable[str] | None = None,
        context_columns: Iterable[str] | None = None,
        expected_columns: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> EIIDataset:
        """Build a dataset from a flat table.

        ``expected_columns`` maps parameter -> boolean column that carries its acquisition
        schedule.
        """
        if time_column is None:
            time = df.index
        else:
            time = df[time_column]
        ctx_cols = list(context_columns or [])
        exp_map = dict(expected_columns or {})
        if parameter_columns is None:
            skip = set(ctx_cols) | set(exp_map.values()) | ({time_column} if time_column else set())
            parameter_columns = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
        params = list(parameter_columns)
        expected = None
        if exp_map:
            expected = pd.DataFrame({p: df[c].to_numpy(dtype=bool) for p, c in exp_map.items()})
        return cls(
            time=time,
            values=df[params],
            context=df[ctx_cols] if ctx_cols else None,
            expected=expected,
            **kwargs,
        )

    # ---------------------------------------------------------------- access
    @property
    def parameters(self) -> list[str]:
        return list(self.values.columns)

    @property
    def n(self) -> int:
        return self.time.n

    def has_parameter(self, name: str) -> bool:
        return name in self.values.columns

    def value(self, parameter: str) -> np.ndarray:
        if parameter not in self.values.columns:
            raise KeyError(f"unknown parameter {parameter!r}; available: {self.parameters}")
        return self.values[parameter].to_numpy(dtype=np.float64)

    def observed(self, parameter: str) -> np.ndarray:
        return ~np.isnan(self.value(parameter))

    def expected_mask(self, parameter: str) -> np.ndarray:
        return self.expected[parameter].to_numpy(dtype=bool)

    def missing_when_expected(self, parameter: str) -> np.ndarray:
        return (~self.observed(parameter)) & self.expected_mask(parameter)

    def context_column(self, name: str) -> np.ndarray:
        if name in self.context.columns:
            return self.context[name].to_numpy()
        if name in self.values.columns:
            return self.value(name)
        raise KeyError(f"unknown context column {name!r}; available: {list(self.context.columns)}")

    def unit(self, parameter: str) -> str:
        return self.units.get(parameter, "")

    # --------------------------------------------------------------- slicing
    def slice(self, start: Any | None = None, end: Any | None = None) -> EIIDataset:
        lo, hi = self.time.slice_indices(start, end)
        return self.iloc(lo, hi)

    def iloc(self, lo: int, hi: int) -> EIIDataset:
        if hi <= lo:
            raise ValueError("empty slice")
        return EIIDataset(
            time=self.time.index[lo:hi],
            values=self.values.iloc[lo:hi],
            context=self.context.iloc[lo:hi] if len(self.context.columns) else None,
            expected=self.expected.iloc[lo:hi],
            quality=self.quality.iloc[lo:hi] if self.quality is not None else None,
            units=self.units,
            events=self.events,
            asset_id=self.asset_id,
            name=self.name,
            attrs=self.attrs,
        )

    def to_frame(self) -> pd.DataFrame:
        df = self.values.copy()
        df.insert(0, "time", self.time.index.to_numpy())
        for c in self.context.columns:
            df[f"ctx__{c}"] = self.context[c].to_numpy()
        for c in self.values.columns:
            df[f"expected__{c}"] = self.expected[c].to_numpy()
        return df

    # -------------------------------------------------------------- manifest
    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.time.t).tobytes())
        for c in self.values.columns:
            h.update(c.encode())
            h.update(np.ascontiguousarray(self.values[c].to_numpy(dtype=np.float64)).tobytes())
            h.update(np.ascontiguousarray(self.expected[c].to_numpy(dtype=bool)).tobytes())
        for c in self.context.columns:
            h.update(c.encode())
            h.update(self.context[c].astype(str).str.cat(sep="\x1f").encode())
        h.update(self.events.content_hash().encode())
        return h.hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "library_version": __version__,
            "name": self.name,
            "asset_id": self.asset_id,
            "n_samples": self.n,
            "time_kind": self.time.kind,
            "time_start": self.time.label(0),
            "time_end": self.time.label(self.n - 1),
            "parameters": self.parameters,
            "context_columns": list(self.context.columns),
            "units": self.units,
            "n_events": len(self.events),
            "content_sha256": self.content_hash(),
            "attrs": self.attrs,
        }

    # ---------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        """Write ``data.csv``, ``events.csv`` and ``manifest.json`` into ``directory``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(d / "data.csv", index=False)
        self.events.to_frame().to_csv(d / "events.csv", index=False)
        self.events.coverage.to_csv(d / "events_coverage.csv", index=False)
        with open(d / "manifest.json", "w") as fh:
            json.dump(self.manifest(), fh, indent=2, default=str)
        return d

    @classmethod
    def load(cls, directory: str | Path, verify: bool = True) -> EIIDataset:
        d = Path(directory)
        with open(d / "manifest.json") as fh:
            man = json.load(fh)
        df = pd.read_csv(d / "data.csv", float_precision="round_trip")
        time = pd.to_datetime(df["time"]) if man["time_kind"] == "datetime" else df["time"]
        params = man["parameters"]
        ctx_cols = man["context_columns"]
        context = df[[f"ctx__{c}" for c in ctx_cols]].rename(columns=lambda c: c[5:]) if ctx_cols else None
        expected = df[[f"expected__{c}" for c in params]].rename(columns=lambda c: c[10:])
        ev_path, cov_path = d / "events.csv", d / "events_coverage.csv"
        coverage = pd.read_csv(cov_path) if cov_path.exists() else None
        events = EventLog.from_frame(pd.read_csv(ev_path), coverage) if ev_path.exists() else EventLog.empty()
        ds = cls(
            time=time,
            values=df[params],
            context=context,
            expected=expected,
            units=man.get("units", {}),
            events=events,
            asset_id=man["asset_id"],
            name=man["name"],
            attrs=man.get("attrs", {}),
        )
        if verify and ds.content_hash() != man["content_sha256"]:
            raise ValueError("dataset content hash does not match manifest; data was modified")
        return ds

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"EIIDataset(name={self.name!r}, asset={self.asset_id!r}, n={self.n}, parameters={self.parameters})"

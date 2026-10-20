"""Provenance helpers: stable identifiers and environment records."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from typing import Any

from .._version import SCHEMA_VERSION, __version__


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def stable_id(*parts: Any, length: int = 16) -> str:
    """Deterministic identifier derived from the canonical JSON of ``parts``."""
    return hashlib.sha256(canonical_json(list(parts)).encode()).hexdigest()[:length]


def config_hash(config: Any) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def provenance_record(**extra: Any) -> dict[str, Any]:
    import matplotlib
    import numpy
    import pandas

    rec = {
        "library": "eii_cloud",
        "library_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
    }
    rec.update(extra)
    return rec

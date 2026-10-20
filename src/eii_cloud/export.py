"""Export layer: SVG / PDF / TIFF / PNG images and JSON sidecars with shared identifiers."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from ._version import __version__
from .evidence.provenance import _json_default, provenance_record

IMAGE_FORMATS = ("svg", "pdf", "tiff", "png")


class ExportManager:
    """Write a figure to several formats with consistent metadata.

    SVG keeps every artist ``gid`` (annotation identifiers appear as ``id="ann-..."``).
    PDF/TIFF are rasterised/vectorised by Matplotlib; the JSON sidecar written by
    :meth:`export_json` is the format-independent record of what was drawn.
    """

    def __init__(self, out_dir: str | Path, dpi: int = 300, transparent: bool = False,
                 metadata: Mapping[str, Any] | None = None, tiff_compression: str = "tiff_lzw"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.transparent = transparent
        self.metadata = dict(metadata or {})
        self.tiff_compression = tiff_compression

    def export(self, fig: Figure, stem: str, formats: Iterable[str] = IMAGE_FORMATS, dpi: int | None = None,
               metadata: Mapping[str, Any] | None = None) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        meta = {**self.metadata, **(metadata or {})}
        dpi = dpi or self.dpi
        for fmt in formats:
            fmt = fmt.lower()
            if fmt not in IMAGE_FORMATS:
                raise ValueError(f"unsupported format {fmt!r}; use one of {IMAGE_FORMATS}")
            path = self.out_dir / f"{stem}.{fmt}"
            kw: dict[str, Any] = {"dpi": dpi, "transparent": self.transparent, "bbox_inches": "tight"}
            if fmt == "svg":
                kw["metadata"] = {"Title": meta.get("title", stem), "Creator": f"eii_cloud {__version__}",
                                  "Description": json.dumps(meta, default=_json_default)}
            elif fmt == "pdf":
                kw["metadata"] = {"Title": meta.get("title", stem), "Creator": f"eii_cloud {__version__}",
                                  "Subject": json.dumps(meta, default=_json_default)[:2000]}
            elif fmt == "png":
                kw["metadata"] = {"Title": meta.get("title", stem), "Software": f"eii_cloud {__version__}",
                                  "Description": json.dumps(meta, default=_json_default)}
            elif fmt == "tiff":
                kw["pil_kwargs"] = {"compression": self.tiff_compression}
            fig.savefig(path, format=fmt, **kw)
            paths[fmt] = path
        return paths

    def export_json(self, obj: Any, stem: str, indent: int = 2) -> Path:
        path = self.out_dir / f"{stem}.json"
        payload = obj.to_dict() if hasattr(obj, "to_dict") else obj
        path.write_text(json.dumps(payload, indent=indent, default=_json_default))
        return path

    def write_manifest(self, stem: str, files: Mapping[str, Path], extra: Mapping[str, Any] | None = None) -> Path:
        man = {"stem": stem, "files": {k: str(Path(v).name) for k, v in files.items()},
               "provenance": provenance_record(), **(extra or {})}
        path = self.out_dir / f"{stem}.manifest.json"
        path.write_text(json.dumps(man, indent=2, default=_json_default))
        return path


def svg_ids(path: str | Path) -> set[str]:
    """All ``id="..."`` attributes of an SVG file (for reproducibility checks)."""
    import re

    text = Path(path).read_text()
    return set(re.findall(r'id="([^"]+)"', text))

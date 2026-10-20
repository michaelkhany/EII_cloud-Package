"""The primary EII Cloud chart."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..data import EIIDataset
from ..findings import AnalysisResult
from ..truth import runs_of
from .annotations import Annotation, build_annotations
from .style import ChartStyle

MODES = ("annotated", "clean", "flags_only", "annotations_json")

DEFAULT_HOWTO = ("How to read this chart:\n"
                 "1. The shaded band is the EII Cloud, i.e., the expected dynamic validity envelope.\n"
                 "2. The solid line with markers is the observed parameter.\n"
                 "3. Instability is interpreted through value deviation, missingness,\n"
                 "    abnormal rigidity, temporal inconsistency, and context contradiction.")


@dataclass
class Layers:
    """Independent on/off switches for the plot layers (section 7.1)."""

    center: bool = True
    bounds: bool = True
    cloud: bool = True
    observed: bool = True
    missing: bool = True
    unsupported: bool = True
    flags: bool = True
    annotations: bool = True
    context: bool = False
    howto: bool = True
    legend: bool = True
    grid: bool = True

    @classmethod
    def for_mode(cls, mode: str, **overrides: Any) -> Layers:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        base = {
            "annotated": cls(),
            "clean": cls(flags=False, annotations=False, howto=False),
            "flags_only": cls(annotations=False, howto=False),
            "annotations_json": cls(),
        }[mode]
        for k, v in overrides.items():
            setattr(base, k, v)
        return base


class EIICloudChart:
    """Observability-aware EII Cloud chart for a single parameter.

    Parameters
    ----------
    dataset, result:
        The analysed data and the analysis result (the chart never recomputes findings).
    parameter:
        Parameter to plot.
    envelope:
        Name of the envelope (in ``result.envelopes``) to draw; defaults to the first
        envelope of the parameter.
    mode:
        ``annotated`` | ``clean`` | ``flags_only`` (``annotations_json`` is handled by
        :func:`~eii_cloud.visualization.annotations.annotations_document`).
    annotation_panel:
        ``True`` places annotation texts in a side panel (dense charts) with numbered flags.
    """

    def __init__(self, dataset: EIIDataset, result: AnalysisResult, parameter: str, envelope: str | None = None,
                 mode: str = "annotated", layers: Layers | None = None, style: ChartStyle | None = None,
                 title: str | None = None, subtitle: str | None = None, howto: str | None = DEFAULT_HOWTO,
                 annotation_panel: bool = False, categories: Iterable[str] | None = None,
                 max_annotations: int | None = None, context_column: str | None = None,
                 interpolate_gaps: bool = False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.dataset, self.result, self.parameter = dataset, result, parameter
        self.mode = mode
        self.layers = layers or Layers.for_mode(mode)
        self.style = style or ChartStyle()
        self.title = title if title is not None else "EII Cloud for a Single Parameter"
        self.subtitle = subtitle if subtitle is not None else f"Observability-aware instability analysis of {parameter}"
        self.howto = howto
        self.annotation_panel = annotation_panel
        self.categories = list(categories) if categories is not None else None
        self.max_annotations = max_annotations
        self.context_column = context_column
        self.interpolate_gaps = interpolate_gaps
        self.envelope_name = envelope or self._default_envelope()
        self.annotations: list[Annotation] = build_annotations(result, dataset, parameter, self.categories)
        if max_annotations is not None:
            self.annotations = self.annotations[:max_annotations]
        self.artists: dict[str, Any] = {}
        self.figure: Figure | None = None
        self.ax: Any = None

    # ------------------------------------------------------------------ data
    def _default_envelope(self) -> str | None:
        for name, e in self.result.envelopes.items():
            if e.get("parameter") == self.parameter:
                return name
        return None

    def _envelope_arrays(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if self.envelope_name is None:
            return None, None, None, None
        e = self.result.envelopes[self.envelope_name]
        lo = np.asarray(e["lower"], dtype=float)
        hi = np.asarray(e["upper"], dtype=float)
        ce = None if e.get("center") is None else np.asarray(e["center"], dtype=float)
        sup = np.asarray(e["support"], dtype=bool)
        return lo, hi, ce, sup

    def _missing_runs(self) -> list[tuple[int, int]]:
        return runs_of(self.dataset.missing_when_expected(self.parameter))

    # ---------------------------------------------------------------- render
    def render(self, ax: Any = None) -> Figure:
        st, L = self.style, self.layers
        if ax is None:
            if self.annotation_panel and L.annotations:
                fig, (ax, panel) = plt.subplots(1, 2, figsize=st.figsize, dpi=st.dpi,
                                                gridspec_kw={"width_ratios": [3, 1.15]})
                panel.axis("off")
            else:
                fig, ax = plt.subplots(figsize=st.figsize, dpi=st.dpi)
                panel = None
        else:
            fig, panel = ax.figure, None
        self.figure, self.ax = fig, ax
        ds, x = self.dataset, self.dataset.time.plot_values()
        y = ds.value(self.parameter)
        lo, hi, ce, sup = self._envelope_arrays()

        if lo is not None:
            if L.cloud:
                band = np.where(sup, True, False)
                art = ax.fill_between(x, np.where(band, lo, np.nan), np.where(band, hi, np.nan), color=st.cloud_color,
                                      alpha=st.cloud_alpha, linewidth=0, label="EII Cloud: expected dynamic validity envelope")
                art.set_gid("eii-cloud")
                self.artists["cloud"] = art
            if L.bounds and not L.cloud:
                ax.plot(x, lo, color=st.cloud_color, lw=1, ls="--", label="Lower validity bound")
                ax.plot(x, hi, color=st.cloud_color, lw=1, ls="--", label="Upper validity bound")
            if L.center and ce is not None:
                (art,) = ax.plot(x, ce, color=st.center_color, lw=st.center_width, label="Expected parameter trajectory")
                art.set_gid("eii-center")
                self.artists["center"] = art
            if L.unsupported and np.any(~sup):
                for s, e in runs_of(~sup):
                    span = ax.axvspan(x[s], x[e], facecolor="none", edgecolor="0.5", hatch=st.unsupported_hatch,
                                      alpha=0.35, lw=0, label="No envelope support" if s == runs_of(~sup)[0][0] else None)
                    span.set_gid(f"eii-unsupported-{s}-{e}")

        if L.observed:
            yy = y.copy()
            if self.interpolate_gaps:
                ok = ~np.isnan(yy)
                yy = np.interp(np.arange(len(yy)), np.flatnonzero(ok), yy[ok])
                lbl = "Observed parameter (gaps interpolated)"
            else:
                lbl = "Observed parameter"
            (art,) = ax.plot(x, yy, color=st.observed_color, lw=st.observed_width, marker=st.marker,
                             ms=st.marker_size, label=lbl)
            art.set_gid("eii-observed")
            self.artists["observed"] = art

        if L.missing:
            first = True
            for s, e in self._missing_runs():
                x0 = x[s - 1] if s > 0 else x[s]
                x1 = x[e + 1] if e + 1 < len(x) else x[e]
                span = ax.axvspan(x0, x1, color=st.missing_color, alpha=st.missing_alpha, lw=0,
                                  label="Missing / unobservable region" if first else None)
                span.set_gid(f"eii-missing-{ds.time.label(s)}-{ds.time.label(e)}")
                self.artists[f"missing-{s}-{e}"] = span
                first = False

        if L.context and self.context_column is not None:
            self._draw_context(ax)

        ymin, ymax = _ylim(y, lo, hi)
        ax.set_ylim(ymin, ymax)
        if L.flags:
            self._draw_flags(ax, y, lo, hi, ce, ymin, ymax)
        if L.annotations and self.mode == "annotated":
            if panel is not None:
                self._draw_panel(panel)
            else:
                self._draw_annotations(ax, y, lo, hi, ce)
        if L.howto and self.howto:
            t = ax.text(0.012, 0.03, self.howto, transform=ax.transAxes, fontsize=st.annotation_size, va="bottom",
                        ha="left", bbox=st.howto_box, zorder=6)
            t.set_gid("eii-howto")
            self.artists["howto"] = t
        unit = ds.unit(self.parameter)
        ax.set_xlabel("Time" if ds.time.is_datetime else "Time window", fontsize=st.font_size)
        ax.set_ylabel("Parameter value" + (f" [{unit}]" if unit else ""), fontsize=st.font_size)
        ax.tick_params(labelsize=st.font_size - 1)
        if L.grid:
            ax.grid(alpha=st.grid_alpha)
        if L.legend:
            ax.legend(loc=st.legend_loc, fontsize=st.font_size - 1, framealpha=0.9)
        ax.set_title(f"{self.title}\n{self.subtitle}", fontsize=st.title_size)
        if ds.time.is_datetime:
            fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # ---------------------------------------------------------------- flags
    def _flag_xy(self, ann: Annotation, y: np.ndarray, lo: Any, hi: Any, ce: Any, ymin: float, ymax: float) -> tuple[Any, float]:
        f = self.result.get(ann.finding_id)
        i = f.anchor_index if f.anchor_index is not None else f.end_index
        xv = self.dataset.time.plot_values()[i]
        if ann.anchor_value is not None and not np.isnan(ann.anchor_value):
            return xv, float(ann.anchor_value)
        if ce is not None and not np.isnan(ce[i]):
            return xv, float(ce[i])
        return xv, ymin + 0.5 * (ymax - ymin)

    def _draw_flags(self, ax: Any, y: np.ndarray, lo: Any, hi: Any, ce: Any, ymin: float, ymax: float) -> None:
        st = self.style
        seen: dict[tuple[Any, float], int] = {}
        for ann in self.annotations:
            xv, yv = self._flag_xy(ann, y, lo, hi, ce, ymin, ymax)
            k = seen.get((xv, yv), 0)            # stagger flags that share an anchor
            seen[(xv, yv)] = k + 1
            marker = st.flag_markers.get(ann.category, "o")
            color = st.flag_colors.get(ann.category, "k")
            art = ax.plot([xv], [yv], marker=marker, ms=11 + 4 * k, mec="black", mfc=color, ls="none",
                          zorder=7 - k, alpha=1.0 if k == 0 else 0.6)[0]
            art.set_gid(f"flag-{ann.annotation_id}")
            self.artists[f"flag-{ann.annotation_id}"] = art
            if self.mode == "flags_only" or self.annotation_panel:
                t = ax.annotate(f"{ann.number}", (xv, yv), xytext=(6, 8 + 13 * k), textcoords="offset points",
                                fontsize=st.annotation_size, fontweight="bold", color=color, zorder=8)
                t.set_gid(f"flagnum-{ann.annotation_id}")
            if ann.start_time != ann.end_time:
                f = self.result.get(ann.finding_id)
                xs = self.dataset.time.plot_values()
                span = ax.axvspan(xs[f.start_index], xs[f.end_index], color=color, alpha=0.07, lw=0, zorder=1)
                span.set_gid(f"interval-{ann.annotation_id}")

    # ---------------------------------------------------------- annotations
    def _draw_annotations(self, ax: Any, y: np.ndarray, lo: Any, hi: Any, ce: Any) -> None:
        """Place annotation boxes with arrows using a collision-free row layout."""
        st = self.style
        ymin, ymax = ax.get_ylim()
        texts, anchors = [], []
        for ann in self.annotations:
            xv, yv = self._flag_xy(ann, y, lo, hi, ce, ymin, ymax)
            t = ax.annotate(ann.text, xy=(xv, yv), xycoords="data", xytext=(0.02, 0.96), textcoords="axes fraction",
                            fontsize=st.annotation_size, va="top", ha="left", bbox=st.annotation_box,
                            arrowprops=st.arrow, zorder=9, annotation_clip=False)
            t.set_gid(ann.annotation_id)
            self.artists[ann.annotation_id] = t
            texts.append(t)
            anchors.append((xv, yv))
        self._layout_annotations(ax, texts, anchors)

    def _layout_annotations(self, ax: Any, texts: list[Any], anchors: list[tuple[Any, float]]) -> None:
        """Pack annotation boxes into rows from the top of the axes, left to right, near their
        anchors, skipping the legend; expand the y-range so rows never cover the data."""
        if not texts:
            return
        fig = ax.figure
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except Exception:  # pragma: no cover - backend without renderer
            return
        ax_bbox = ax.get_window_extent(renderer)
        W, H = max(ax_bbox.width, 1.0), max(ax_bbox.height, 1.0)

        def to_frac(bb):
            return ((bb.x0 - ax_bbox.x0) / W, (bb.y0 - ax_bbox.y0) / H, (bb.x1 - ax_bbox.x0) / W, (bb.y1 - ax_bbox.y0) / H)

        sizes = []
        for t in texts:
            patch = t.get_bbox_patch()
            bb = patch.get_window_extent(renderer) if patch is not None else t.get_window_extent(renderer)
            sizes.append((bb.width / W + 0.012, bb.height / H + 0.018))
        legend = ax.get_legend()
        leg = to_frac(legend.get_window_extent(renderer)) if legend is not None else None
        ax_frac = []
        for xv, yv in anchors:
            px, py = ax.transData.transform((ax.convert_xunits(xv), ax.convert_yunits(yv)))
            ax_frac.append(((px - ax_bbox.x0) / W, (py - ax_bbox.y0) / H))
        order = sorted(range(len(texts)), key=lambda i: ax_frac[i][0])
        y_top, cursor, row_h, pad = 0.975, 0.012, 0.0, 0.012
        lowest = y_top

        def start_x(desired: float, w: float, top: float, h: float) -> float:
            x = max(cursor, desired)
            if leg is not None and not (top - h > leg[3] or top < leg[1]) and x < leg[2] + pad and x + w > leg[0]:
                x = max(x, leg[2] + pad)
            return x

        for i in order:
            w, h = sizes[i]
            desired = float(np.clip(ax_frac[i][0] - w / 2, pad, 1 - pad - w))
            x = start_x(desired, w, y_top, h)
            if x + w > 1 - pad + 1e-9:
                y_top -= row_h + pad
                cursor, row_h = 0.012, 0.0
                x = start_x(desired, w, y_top, h)
                if x + w > 1 - pad + 1e-9:
                    x = max(pad, 1 - pad - w)
            texts[i].set_position((x, y_top))
            cursor = x + w + pad
            row_h = max(row_h, h)
            lowest = min(lowest, y_top - h)
        # keep every annotation box clear of the data *below it* by expanding the upper y-range
        placed = []
        for i in order:
            x, top = texts[i].get_position()
            w, h = sizes[i]
            placed.append((x, x + w, top - h))
        scale = self._required_scale(ax, placed, ax_bbox)
        if scale > 1.0:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymin + (ymax - ymin) * scale)

    def _required_scale(self, ax: Any, boxes: list[tuple[float, float, float]], ax_bbox: Any) -> float:
        """Smallest factor by which the y-range must grow so that each box's bottom edge stays
        above the highest data/envelope value within the box's horizontal extent."""
        ymin, ymax = ax.get_ylim()
        if ymax <= ymin:
            return 1.0
        y = self.dataset.value(self.parameter)
        lo, hi, ce, sup = self._envelope_arrays()
        xmin_pix, xmax_pix = ax_bbox.x0, ax_bbox.x1
        # map sample times to axes fractions once
        px = np.array([ax.transData.transform((ax.convert_xunits(v), 0.0))[0] for v in self.dataset.time.plot_values()])
        fx = (px - xmin_pix) / max(xmax_pix - xmin_pix, 1.0)
        scale = 1.0
        for x0, x1, bottom in boxes:
            m = (fx >= x0 - 0.01) & (fx <= x1 + 0.01)
            vals = []
            if m.any():
                if np.any(~np.isnan(y[m])):
                    vals.append(float(np.nanmax(y[m])))
                if hi is not None and np.any(sup[m]):
                    vals.append(float(np.nanmax(hi[m & sup])))
            if not vals:
                continue
            top_frac = (max(vals) - ymin) / (ymax - ymin)
            need = top_frac / max(bottom - 0.03, 0.2)
            scale = max(scale, need)
        return float(min(scale, 2.5))

    def _draw_panel(self, panel: Any) -> None:
        st = self.style
        lines = []
        for ann in self.annotations:
            lines.append(f"[{ann.number}] {ann.category_code} {ann.short_label}  ({ann.start_time} .. {ann.end_time})\n"
                         + "\n".join("    " + ln for ln in ann.text.splitlines()))
        t = panel.text(0.0, 1.0, "\n\n".join(lines), transform=panel.transAxes, fontsize=st.annotation_size, va="top",
                       ha="left", family="DejaVu Sans")
        t.set_gid("eii-annotation-panel")
        for ann in self.annotations:
            self.artists[ann.annotation_id] = t  # the panel text carries the ids
        panel.set_title("Annotations", fontsize=st.font_size)

    def _draw_context(self, ax: Any) -> None:
        col = np.asarray(self.dataset.context_column(self.context_column))
        x = self.dataset.time.plot_values()
        ymin, ymax = ax.get_ylim()
        if col.dtype.kind in "fiu":
            ax2 = ax.twinx()
            ax2.plot(x, col.astype(float), color="0.4", lw=1, ls=":", label=self.context_column)
            ax2.set_ylabel(self.context_column, color="0.4")
            return
        cats = sorted({str(c) for c in col})
        cmap = plt.get_cmap("Pastel2")
        for k, cat in enumerate(cats):
            for s, e in runs_of(col.astype(str) == cat):
                ax.axvspan(x[s], x[e], color=cmap(k % 8), alpha=0.25, lw=0, zorder=0,
                           label=f"{self.context_column}={cat}" if s == runs_of(col.astype(str) == cat)[0][0] else None)

    # -------------------------------------------------------------- exports
    def annotation_ids(self) -> list[str]:
        return [a.annotation_id for a in self.annotations]


def _ylim(y: np.ndarray, lo: Any, hi: Any) -> tuple[float, float]:
    parts = [y]
    if lo is not None:
        parts += [lo, hi]
    allv = np.concatenate([np.asarray(p, dtype=float) for p in parts])
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(allv.min()), float(allv.max())
    pad = 0.08 * (vmax - vmin if vmax > vmin else 1.0)
    return vmin - pad, vmax + pad * 1.6

"""Synchronized vertical cursor with hover readout for matplotlib axes.

Provides a ``SyncedCursor`` that attaches a vertical line (matching the mouse
x-position) to every registered axis and displays an annotation per axis with
the values of each line at the cursor's time instant. Uses blitting to keep
interaction responsive on dense time-series without redrawing the whole figure.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

__all__ = ["SyncedCursor"]


class SyncedCursor:
    """Cross-axes vertical cursor with per-axis value readout.

    The cursor is attached to a list of matplotlib ``Axes`` that share a time
    x-axis. Moving the mouse over any of them draws a vertical line at the same
    x value in all axes and updates a text annotation with the interpolated
    values of every line in each axis.
    """

    def __init__(
        self,
        fig: Figure,
        axes: Iterable[Axes],
        *,
        show_readout: bool = True,
        line_kwargs: dict | None = None,
        text_kwargs: dict | None = None,
    ) -> None:
        self.fig = fig
        self.axes: list[Axes] = [ax for ax in axes if ax is not None]
        self.show_readout = show_readout

        self._line_kwargs = {"color": "0.35", "linewidth": 0.8, "linestyle": ":"}
        if line_kwargs:
            self._line_kwargs.update(line_kwargs)

        self._text_kwargs = {
            "fontsize": 8,
            "color": "black",
            "bbox": {"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.5", "alpha": 0.85},
            "verticalalignment": "top",
        }
        if text_kwargs:
            self._text_kwargs.update(text_kwargs)

        self._vlines = []
        self._texts = []
        for ax in self.axes:
            vline = ax.axvline(x=np.nan, animated=True, visible=False, **self._line_kwargs)
            txt = ax.annotate(
                "",
                xy=(0.99, 0.98),
                xycoords="axes fraction",
                animated=True,
                visible=False,
                **self._text_kwargs,
            )
            txt.set_horizontalalignment("right")
            self._vlines.append(vline)
            self._texts.append(txt)

        self._backgrounds = [None] * len(self.axes)
        self._cid_move = fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self._cid_draw = fig.canvas.mpl_connect("draw_event", self._on_draw)
        self._cid_leave = fig.canvas.mpl_connect("axes_leave_event", self._on_leave)

    def disconnect(self) -> None:
        """Detach all matplotlib callbacks."""
        for cid in (self._cid_move, self._cid_draw, self._cid_leave):
            self.fig.canvas.mpl_disconnect(cid)

    def _on_draw(self, _event) -> None:
        for i, ax in enumerate(self.axes):
            self._backgrounds[i] = self.fig.canvas.copy_from_bbox(ax.bbox)

    def _on_leave(self, _event) -> None:
        for vline, txt in zip(self._vlines, self._texts):
            vline.set_visible(False)
            txt.set_visible(False)
        self._blit()

    def _on_move(self, event: MouseEvent) -> None:
        if event.inaxes not in self.axes or event.xdata is None:
            return
        x = float(event.xdata)
        for ax, vline, txt in zip(self.axes, self._vlines, self._texts):
            vline.set_xdata([x, x])
            vline.set_visible(True)
            if self.show_readout:
                txt.set_text(self._build_readout(ax, x))
                txt.set_visible(True)
            else:
                txt.set_visible(False)
        self._blit()

    def _blit(self) -> None:
        for ax, bg, vline, txt in zip(self.axes, self._backgrounds, self._vlines, self._texts):
            if bg is None:
                continue
            self.fig.canvas.restore_region(bg)
            ax.draw_artist(vline)
            if txt.get_visible():
                ax.draw_artist(txt)
            self.fig.canvas.blit(ax.bbox)

    def _build_readout(self, ax: Axes, x: float) -> str:
        lines = [f"t = {x:.4g}"]
        for line in ax.get_lines():
            if line is self._vlines[self.axes.index(ax)]:
                continue
            xs = np.asarray(line.get_xdata(), dtype=float)
            ys = np.asarray(line.get_ydata(), dtype=float)
            if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
                continue
            if x < xs[0] or x > xs[-1]:
                continue
            idx = int(np.searchsorted(xs, x))
            if idx <= 0:
                y = float(ys[0])
            elif idx >= xs.size:
                y = float(ys[-1])
            else:
                x0, x1 = xs[idx - 1], xs[idx]
                if x1 == x0:
                    y = float(ys[idx])
                else:
                    t = (x - x0) / (x1 - x0)
                    y = float(ys[idx - 1] + t * (ys[idx] - ys[idx - 1]))
            label = line.get_label()
            if not label or label.startswith("_"):
                continue
            lines.append(f"{label}: {y:.4g}")
        return "\n".join(lines)

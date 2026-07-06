"""Matplotlib figure layout for signal plots — self-contained styling, no App import."""

from __future__ import annotations

import traceback

import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

DEFAULT_AXIS_KEYS = ("top", "mid", "bot")
DEFAULT_XLIM = (0.0, 2.5)
DEFAULT_YLIMS = {"top": (-1.0, 1.0), "mid": (-1.0, 1.0), "bot": (-10.0, 10.0)}


class SignalFigureHost:
    """Three-row signal figure embedded in a tk parent frame."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        axis_keys: tuple[str, ...] = DEFAULT_AXIS_KEYS,
        on_canvas_created=None,
    ) -> None:
        self.parent = parent
        self.axis_keys = axis_keys
        self.on_canvas_created = on_canvas_created
        self.fig = None
        self.axes: dict[str, plt.Axes] = {}
        self.canvas: FigureCanvasTkAgg | None = None
        self.ccs: dict[str, tk.Canvas] = {}
        self._fig_nrows = len(axis_keys)
        self.build()

    def build(self) -> None:
        for widget in self.parent.winfo_children():
            widget.destroy()
        prev = self.fig
        if prev is not None:
            try:
                plt.close(prev)
            except Exception:
                traceback.print_exc()

        nrows = len(self.axis_keys)
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.05)
        self.fig.clf()
        self.axes = {}
        for r, key in enumerate(self.axis_keys):
            ax = self.fig.add_subplot(nrows, 1, r + 1)
            ax.set_xlim(*DEFAULT_XLIM)
            ylo, yhi = DEFAULT_YLIMS.get(key, (-1.0, 1.0))
            ax.set_ylim(ylo, yhi)
            ax.set_autoscale_on(False)
            self.axes[key] = ax
        self._fig_nrows = nrows
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().grid(column=0, row=0, rowspan=nrows, sticky=tk.NSEW)
        if self.on_canvas_created is not None:
            self.on_canvas_created(self.canvas)
        self.build_legend_canvases()

    def build_legend_canvases(self) -> None:
        self.ccs = {}
        for r, key in enumerate(self.axes.keys()):
            cc = tk.Canvas(
                self.parent,
                width=110,
                height=130,
                bg="white",
                highlightbackground="white",
            )
            cc.grid(column=1, row=r, sticky="")
            self.ccs[key] = cc
        nrows = len(self.axes)
        for r in range(nrows):
            self.parent.grid_rowconfigure(r, weight=1)
        self.parent.grid_columnconfigure(0, weight=6)
        self.parent.grid_columnconfigure(1, weight=1)

    def attach_to(self, target) -> None:
        """Copy figure handles onto an app-like object (mediator glue)."""
        target.fig = self.fig
        target.axes = self.axes
        target.canvas = self.canvas
        target.ccs = self.ccs
        target._fig_nrows = self._fig_nrows

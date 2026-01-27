from __future__ import annotations
# App/Area.py
import tkinter as tk
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .App import App

from matplotlib.axes import Axes
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle

class Area:
    def __init__(self, app:App, index:int, key:str, kind="stim",
                 t0_s=0.0, t1_s=0.5, color="green", fs_hz=1000.0):
        self.app = app
        self.index = int(index)
        self.key = str(key)
        self.kind = str(kind)
        self.color = color
        self.fs = float(fs_hz)

        self.t = [float(t0_s), float(t1_s)]  # seconds

        self.ax: Axes | None = None
        self.canvas = None
        self.line0 = None
        self.line1 = None
        self.rect = None
        self.ylim = None

    def attach(self, ax: Axes) -> None:
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.ylim = list(ax.get_ylim())

        y0, y1 = self.ylim
        t0, t1 = self.t

        # two vertical draggable boundaries (we'll add dragging later)
        self.line_L = mlines.Line2D([t0, t0], [y0, y1], linewidth=0.8, color="black", alpha=0.35)
        self.line_R = mlines.Line2D([t1, t1], [y0, y1], linewidth=0.8, color="black", alpha=0.35)

        # shaded region
        left = min(t0, t1)
        self.rect = Rectangle(
            (left, y0),
            width=abs(t1 - t0),
            height=(y1 - y0),
            facecolor=self.color,
            alpha=0.20,
            edgecolor="none",
        )
        self.line_L.set_picker(2)
        self.line_R.set_picker(2)

        ax.add_line(self.line_L)
        ax.add_line(self.line_R)
        ax.add_patch(self.rect)

        self.canvas.draw_idle()
        self.connect()

    def connect(self) -> None:

        active_line = None      # "left" or "right"
        cid_pick = None
        cid_follower = None
        cid_releaser = None
        def on_release(event):
            nonlocal cid_follower,cid_releaser
            # end drag on ANY release (prevents stacking)
            # assign delta, disconnect, then update_plot
            t0, t1 = sorted(self.t)
            s0 = int(round(t0 * self.fs))
            s1 = int(round(t1 * self.fs))

            # assign ONLY if slot exists (no extending/adding)
            pidx = self.app.to_index[self.app.i][self.app.j]
            lst = self.app.delta[pidx][2][self.kind]

            lst[self.index] = [s0, s1]
            tv = self.app.table.tree  # EditableTree
            row = pidx
            iid = f"row{row}"
            vals = list(tv.item(iid)["values"])
            c1=self.app.delta[pidx][2]
            vals[-1] = ", ".join(
                f"{k}: {', '.join(map(str, v))}"
                for k, v in c1.items()
                if "voltage" not in k
            )
            tv.item(iid, values=vals)
            self.canvas.mpl_disconnect(cid_follower)
            self.canvas.mpl_disconnect(cid_releaser)
            active_line = None      # "left" or "right"
            cid_pick = None
            cid_follower = None
            cid_releaser = None
            self.canvas.draw_idle()
            self.app.update_plot()

        def on_follow(event):
            nonlocal active_line
            if active_line is None:
                return
            if event.inaxes is not self.ax or event.xdata is None:
                return

            if active_line == "left":
                self.line_L.set_xdata([event.xdata, event.xdata])
                self.t[0] = event.xdata
            elif active_line == "right":
                self.line_R.set_xdata([event.xdata, event.xdata])
                self.t[1] = event.xdata

            # update rectangle every frame
            t0, t1 = self.t
            y0, y1 = self.ylim

            left = min(t0, t1)
            width = abs(t1 - t0)

            self.rect.set_x(left)
            self.rect.set_width(width)

            # color rule: if right < left -> red + alpha 0.6 else normal color + alpha 0.2
            if t1 < t0:
                self.rect.set_facecolor("red")
                self.rect.set_alpha(0.6)
            else:
                self.rect.set_facecolor(self.color)
                self.rect.set_alpha(0.2)

            self.canvas.draw_idle()
            self.app.update_idletasks()

        def on_pick(event):
            nonlocal active_line, cid_follower,cid_releaser
            me = event.mouseevent
            if me is None or me.button != 1:
                return

            if event.artist is self.line_L:
                active_line = "left"
            elif event.artist is self.line_R:
                active_line = "right"
            else:
                return

            cid_follower = self.canvas.mpl_connect("motion_notify_event", on_follow)
            cid_releaser = self.canvas.mpl_connect("button_release_event", on_release)

        cid_pick = self.canvas.mpl_connect("pick_event", on_pick)



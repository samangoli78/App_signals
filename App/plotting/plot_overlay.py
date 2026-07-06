"""Matplotlib shade rectangles and draggable signal-area overlays."""

from __future__ import annotations

import matplotlib.lines as lines
from matplotlib.patches import Rectangle


class Shades:
    def __init__(self, app) -> None:
        self.app = app
        self.rectangle = 0

    def add_shade(self, x=(0, 1), y=(-1, 1), config=False, **kwargs):
        self.ylim = y
        for key, value in kwargs.items():
            if key == "color":
                self.color_init = value
        if (x[1] - x[0]) < 0:
            self.color = "red"
            if config:
                self.rectangle.set_xy([x[1], y[0]])
                self.rectangle.set_height(y[1] - y[0])
                self.rectangle.set_width(x[0] - x[1])
                self.rectangle.set(color=self.color)
            else:
                rectangle = Rectangle([x[1], y[0]], width=x[0] - x[1], height=y[1] - y[0], color=self.color, alpha=0.2)
                self.rectangle = rectangle
        else:
            self.color = self.color_init
            if config:
                self.rectangle.set_xy([x[0], y[0]])
                self.rectangle.set_height(y[1] - y[0])
                self.rectangle.set_width(x[1] - x[0])
                self.rectangle.set(color=self.color)
            else:
                rectangle = Rectangle([x[0], y[0]], width=x[1] - x[0], height=y[1] - y[0], color=self.color, alpha=0.2)
                self.rectangle = rectangle

    def plot(self, ax):
        ax.add_artist(self.rectangle)


class Area(Shades):
    def __init__(self, app, ind) -> None:
        super().__init__(app)
        self.index = ind
        self.app = app

    def add_area(self, arg, ylim: tuple[2], xlim: tuple[2], color):
        lines_ = []
        self.ylim = ylim
        self.xlim = xlim
        self.arg_sample = arg
        arg = arg * 0.001
        self.x = arg
        lines_.append(self.define_line(x0=self.x[0], x1=self.x[0], y0=ylim[0], y1=ylim[1]))
        lines_.append(self.define_line(x0=self.x[1], x1=self.x[1], y0=ylim[0], y1=ylim[1]))
        self.add_shade(x=self.x, y=ylim, color=color)
        self.color = color
        self.lines = lines_

    def plot_area(self, ax):
        self.ax = ax
        ax.add_line(self.lines[0])
        ax.add_line(self.lines[1])
        super().plot(ax)

    def define_line(self, x0, x1, y0, y1):
        x = [x0, x1]
        y = [y0, y1]
        line = lines.Line2D(x, y, picker=3, linewidth=0.1, color="black", alpha=0.3)
        return line

    def configure_shade_attr(self, x, y, step_x=0.1, step_y=0.1, **kwargs):
        for key, value in kwargs.items():
            if key == "color":
                self.color_init_ = value
        self.add_shade(x=x, y=y, config=True)

    def clickonline(self, event=None):
        if event.mouseevent.button != 1:
            return
        for iter_, line in enumerate(self.lines):
            if event.artist == line:
                self.selected_line = line
                self.iter = iter_
                self.follower = self.app.canvas.mpl_connect("motion_notify_event", self.followmouse)
                self.releaser = self.app.canvas.mpl_connect("button_release_event", self.releaseonclick)

    def followmouse(self, event=None):
        self.selected_line.set_xdata([event.xdata, event.xdata])
        self.x[self.iter] = event.xdata
        self.configure_shade_attr(x=self.x, y=self.ylim)
        self.app.canvas.draw_idle()

    def releaseonclick(self, event=None):
        self.app.canvas.mpl_disconnect(self.releaser)
        self.app.canvas.mpl_disconnect(self.follower)
        self.arg_second = self.x
        self.arg_sample = self.arg_second * 1000
        key = "stim" if self.color == "green" else "sinus"
        update = [int(arg) for arg in self.arg_sample]
        if update[1] >= update[0]:
            self.app.delta[self.app.to_index[self.app.i][self.app.j]][2][key][self.index] = update
            self.app.update_plot()

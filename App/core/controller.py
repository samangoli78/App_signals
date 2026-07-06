"""Point/section navigation for the main app."""

from __future__ import annotations


class AppController:
    """Keyboard and ribbon navigation over ``app.i`` / ``app.j`` indices."""

    def __init__(self, app) -> None:
        self.app = app

    def p_increase(self, event=None):
        self.app.direction = 1
        if self.app.j >= len(self.app.cont[self.app.i][0]) - 1:
            self.s_increase(None)
            self.app.j = 0
            self.app.update_plot()
        else:
            self.app.j += 1
            self.app.update_plot()

    def p_decrease(self, event=None):
        self.app.direction = -1
        if self.app.j <= 0:
            self.s_decrease(None)
            self.app.j = len(self.app.cont[self.app.i][0]) - 1
            self.app.update_plot()
        else:
            self.app.j -= 1
            self.app.update_plot()

    def s_increase(self, event=None):
        self.app.direction = 1
        if self.app.i >= len(self.app.cont) - 1:
            self.app.i = 0
        else:
            self.app.i += 1
        if event:
            self.app.j = 0
            self.app.update_plot()

    def s_decrease(self, event=None):
        self.app.direction = -1
        if self.app.i <= 0:
            self.app.i = len(self.app.cont) - 1
        else:
            self.app.i -= 1
        if event:
            self.app.j = 0
            self.app.update_plot()

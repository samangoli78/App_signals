from __future__ import annotations
# Future annotations:
# postpone type-hint evaluation, which helps avoid import cycles and speeds imports.

from .base_components import BaseController

class AppController(BaseController):
    # Controller concept:
    # centralizes navigation/state transitions so UI code does not duplicate movement rules.
    """Navigation and state transitions for the main app."""

    def __init__(self, app) -> None:
        # Dependency injection:
        # controller receives the app instance instead of constructing it.
        super().__init__(app)

    def p_increase(self, event=None):
        # Wrap-around navigation:
        # when the point index reaches the end, move to next section and restart at point 0.
        self.app.direction = 1
        if self.app.j >= len(self.app.cont[self.app.i][0]) - 1:
            self.s_increase(None)
            self.app.j = 0
            self.app.update_plot()
        else:
            self.app.j += 1
            self.app.update_plot()

    def p_decrease(self, event=None):
        # Reverse wrap-around:
        # when index goes below 0, move to previous section and jump to its last point.
        self.app.direction = -1
        if self.app.j <= 0:
            self.s_decrease(None)
            self.app.j = len(self.app.cont[self.app.i][0]) - 1
            self.app.update_plot()
        else:
            self.app.j -= 1
            self.app.update_plot()

    def s_increase(self, event=None):
        # Optional event parameter lets this method work from both keyboard events and direct calls.
        self.app.direction = 1
        if self.app.i >= len(self.app.cont) - 1:
            self.app.i = 0
        else:
            self.app.i += 1
        # Conditional side effect: only reset point when navigation came from a section-level event.
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


class SyncingAppController(AppController):
    """Extension point for future sync-aware navigation hooks."""

    def _after_navigation(self):
        # Hook for future session/network sync logic.
        return None

    def p_increase(self, event=None):
        out = super().p_increase(event)
        self._after_navigation()
        return out

    def p_decrease(self, event=None):
        out = super().p_decrease(event)
        self._after_navigation()
        return out

    def s_increase(self, event=None):
        out = super().s_increase(event)
        self._after_navigation()
        return out

    def s_decrease(self, event=None):
        out = super().s_decrease(event)
        self._after_navigation()
        return out

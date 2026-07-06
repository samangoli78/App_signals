"""Pause OpenGL redraws while the 3D viewer ribbon / panes are resized."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

_SASH_HIT_PX = 10
_DEBOUNCE_MS = 120


def _pane_orientation(pane: ttk.PanedWindow) -> str:
    try:
        return str(pane.cget("orient"))
    except tk.TclError:
        return "vertical"


def _near_sash(pane: ttk.PanedWindow, event) -> bool:
    try:
        pos = pane.sashpos(0)
    except tk.TclError:
        return False
    if _pane_orientation(pane) == "horizontal":
        return abs(event.x - pos) <= _SASH_HIT_PX
    return abs(event.y - pos) <= _SASH_HIT_PX


def attach_viewer_resize_pause(viewer, panes, watch_widgets=(), debounce_ms=_DEBOUNCE_MS) -> None:
    """Pause only the OpenGL viewer during its ribbon / pane resize."""
    state = {"dragging": False, "configure_after": None, "sizes": {}}

    def _schedule_end() -> None:
        if state["configure_after"] is not None:
            try:
                viewer.after_cancel(state["configure_after"])
            except Exception:
                pass
        state["configure_after"] = viewer.after(debounce_ms, _end_pause)

    def _end_pause() -> None:
        state["configure_after"] = None
        state["dragging"] = False
        viewer.set_render_paused(False)

    def _begin_pause() -> None:
        viewer.set_render_paused(True)

    def _on_sash_press(event) -> None:
        pane = event.widget
        if isinstance(pane, ttk.PanedWindow) and _near_sash(pane, event):
            state["dragging"] = True
            _begin_pause()

    def _on_sash_motion(event) -> None:
        pane = event.widget
        if not isinstance(pane, ttk.PanedWindow):
            return
        if state["dragging"] or _near_sash(pane, event):
            if not state["dragging"]:
                state["dragging"] = True
                _begin_pause()
            _schedule_end()

    def _on_sash_release(_event) -> None:
        if state["dragging"]:
            _schedule_end()

    def _on_configure(event) -> None:
        widget = event.widget
        key = id(widget)
        size = (event.width, event.height)
        previous = state["sizes"].get(key)
        state["sizes"][key] = size
        if previous is None or previous == size:
            return
        _begin_pause()
        _schedule_end()

    for pane in panes:
        pane.bind("<ButtonPress-1>", _on_sash_press, add="+")
        pane.bind("<B1-Motion>", _on_sash_motion, add="+")
        pane.bind("<ButtonRelease-1>", _on_sash_release, add="+")

    for widget in watch_widgets:
        if widget is not None:
            widget.bind("<Configure>", _on_configure, add="+")

"""Pause matplotlib + OpenGL redraws while panes are being resized."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

_SASH_HIT_PX = 10
_DEBOUNCE_MS = 120


def _mesh_viewer(app):
    panel = getattr(app, "mesh_panel", None)
    if panel is None:
        return None
    return getattr(panel, "viewer", None)


def begin_resize_pause(app) -> None:
    depth = getattr(app, "_resize_pause_depth", 0) + 1
    app._resize_pause_depth = depth
    if depth > 1:
        return
    app._resize_paused = True
    viewer = _mesh_viewer(app)
    if viewer is not None:
        viewer.set_render_paused(True)


def end_resize_pause(app) -> None:
    depth = max(0, getattr(app, "_resize_pause_depth", 0) - 1)
    app._resize_pause_depth = depth
    if depth > 0:
        return
    app._resize_paused = False
    viewer = _mesh_viewer(app)
    if viewer is not None:
        viewer.set_render_paused(False)
    if getattr(app, "_pending_canvas_draw", False):
        app._pending_canvas_draw = False
        canvas = getattr(app, "canvas", None)
        if canvas is None:
            return
        original = getattr(canvas, "_original_draw_idle", None)
        if original is not None:
            original()
        else:
            canvas.draw_idle()


def install_canvas_draw_guard(app) -> None:
    canvas = getattr(app, "canvas", None)
    if canvas is None:
        return
    if getattr(app, "_draw_guard_canvas", None) is canvas:
        return
    app._draw_guard_canvas = canvas
    original = canvas.draw_idle
    canvas._original_draw_idle = original
    canvas._draw_guard_installed = True
    app._pending_canvas_draw = False

    def guarded_draw_idle():
        if getattr(app, "_resize_paused", False):
            app._pending_canvas_draw = True
            return
        app._pending_canvas_draw = False
        original()

    canvas.draw_idle = guarded_draw_idle


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


def attach_resize_pause(app, panes, watch_widgets=(), debounce_ms=_DEBOUNCE_MS) -> None:
    """Pause plot + 3D rendering during sash drags and size changes."""
    app._resize_paused = False
    app._resize_pause_depth = 0
    app._pending_canvas_draw = False

    state = {"dragging": False, "configure_after": None, "sizes": {}}

    def _schedule_end() -> None:
        if state["configure_after"] is not None:
            try:
                app.after_cancel(state["configure_after"])
            except Exception:
                pass
        state["configure_after"] = app.after(debounce_ms, _end_pause)

    def _end_pause() -> None:
        state["configure_after"] = None
        state["dragging"] = False
        end_resize_pause(app)

    def _on_sash_press(event) -> None:
        pane = event.widget
        if isinstance(pane, ttk.PanedWindow) and _near_sash(pane, event):
            state["dragging"] = True
            begin_resize_pause(app)

    def _on_sash_motion(event) -> None:
        pane = event.widget
        if not isinstance(pane, ttk.PanedWindow):
            return
        if state["dragging"] or _near_sash(pane, event):
            if not state["dragging"]:
                state["dragging"] = True
                begin_resize_pause(app)
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
        if not getattr(app, "_resize_paused", False):
            begin_resize_pause(app)
        _schedule_end()

    for pane in panes:
        pane.bind("<ButtonPress-1>", _on_sash_press, add="+")
        pane.bind("<B1-Motion>", _on_sash_motion, add="+")
        pane.bind("<ButtonRelease-1>", _on_sash_release, add="+")

    for widget in watch_widgets:
        if widget is not None:
            widget.bind("<Configure>", _on_configure, add="+")


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

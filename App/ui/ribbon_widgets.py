"""Shared collapsible ribbon widgets for main app and 3D viewer."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

RIBBON_BG = "#4a4a4a"
RIBBON_HEADER_BG = "#3a3a3a"
SECTION_BG = "#525252"
RIBBON_FONT = ("timesnewroman", 10)
MIN_EXPANDED_WIDTH = 120


def _clamp_ribbon_sash(outer_pane, ribbon_open, collapsed_width) -> None:
    """Keep the ribbon pane visible: collapsed strip or a usable expanded width."""
    cw = int(collapsed_width["px"])
    try:
        pos = int(outer_pane.sashpos(0))
    except tk.TclError:
        return
    if not ribbon_open.get():
        if pos != cw:
            try:
                outer_pane.sashpos(0, cw)
            except tk.TclError:
                pass
        return
    if pos <= cw + 8:
        return "collapse"
    if pos < MIN_EXPANDED_WIDTH:
        try:
            outer_pane.sashpos(0, MIN_EXPANDED_WIDTH)
        except tk.TclError:
            pass


def ribbon_button(parent, text, command, **kwargs):
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=SECTION_BG,
        fg="white",
        activebackground="#666666",
        activeforeground="white",
        relief=tk.FLAT,
        font=RIBBON_FONT,
        anchor="w",
        **kwargs,
    )
    btn.pack(fill="x", padx=6, pady=2)
    return btn


def ribbon_label(parent, text, *, bold=False, wraplength=None):
    font = (RIBBON_FONT[0], RIBBON_FONT[1], "bold") if bold else RIBBON_FONT
    kw = dict(
        text=text,
        bg=RIBBON_BG,
        fg="white",
        font=font,
        anchor="w",
    )
    if wraplength is not None:
        kw["wraplength"] = wraplength
        kw["justify"] = "left"
    label = tk.Label(parent, **kw)
    label.pack(fill="x", padx=6, pady=(2, 4))
    return label


def ribbon_checkbox(parent, text, variable, command=None):
    cb = tk.Checkbutton(
        parent,
        variable=variable,
        command=command,
        text=text,
        font=RIBBON_FONT,
        bg=RIBBON_BG,
        fg="white",
        activebackground=RIBBON_BG,
        activeforeground="white",
        selectcolor="#333333",
        anchor="w",
    )
    cb.pack(fill="x", padx=6, pady=2)
    return cb


def collapsible_section(parent, title, expanded=False):
    outer = tk.Frame(parent, bg=RIBBON_BG)
    outer.pack(fill="x", padx=4, pady=3)

    header = tk.Frame(outer, bg=RIBBON_HEADER_BG)
    header.pack(fill="x")

    is_open = tk.BooleanVar(value=expanded)
    toggle = tk.Button(
        header,
        text="▼" if expanded else "▶",
        width=2,
        relief=tk.FLAT,
        bg=RIBBON_HEADER_BG,
        fg="white",
        activebackground=RIBBON_HEADER_BG,
        font=RIBBON_FONT,
    )
    toggle.pack(side=tk.LEFT, padx=(2, 0))

    tk.Label(
        header,
        text=title,
        bg=RIBBON_HEADER_BG,
        fg="white",
        font=(RIBBON_FONT[0], RIBBON_FONT[1], "bold"),
        anchor="w",
    ).pack(side=tk.LEFT, fill="x", expand=True, padx=4, pady=3)

    body = tk.Frame(outer, bg=RIBBON_BG)
    if expanded:
        body.pack(fill="x", padx=4, pady=(0, 4))

    def _toggle():
        if is_open.get():
            body.pack_forget()
            toggle.config(text="▶")
            is_open.set(False)
        else:
            body.pack(fill="x", padx=4, pady=(0, 4))
            toggle.config(text="▼")
            is_open.set(True)

    toggle.config(command=_toggle)
    return {"outer": outer, "body": body, "toggle": toggle}


def make_scrollable_column(parent):
    canvas = tk.Canvas(parent, bg=RIBBON_BG, highlightthickness=0, borderwidth=0)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=RIBBON_BG)

    inner.bind(
        "<Configure>",
        lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    def _on_canvas_configure(event):
        canvas.itemconfigure(window_id, width=event.width)

    canvas.bind("<Configure>", _on_canvas_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
    canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    return canvas, inner


def build_ribbon_shell(parent, *, title="Tools", default_width=240, start_collapsed=True):
    """Horizontal split: resizable collapsible left ribbon | content host."""
    outer_pane = ttk.PanedWindow(parent, orient="horizontal")
    outer_pane.pack(fill=tk.BOTH, expand=True)

    ribbon_shell = tk.Frame(outer_pane, bg=RIBBON_BG, width=default_width)
    outer_pane.add(ribbon_shell, weight=0)

    ribbon_header = tk.Frame(ribbon_shell, bg=RIBBON_HEADER_BG)
    ribbon_header.pack(fill="x")

    ribbon_open = tk.BooleanVar(value=not start_collapsed)
    last_width = {"px": default_width}
    collapsed_width = {"px": 32}

    ribbon_toggle = tk.Button(
        ribbon_header,
        text="▶" if start_collapsed else "◀",
        width=2,
        relief=tk.FLAT,
        bg=RIBBON_HEADER_BG,
        fg="white",
        activebackground=RIBBON_HEADER_BG,
        font=RIBBON_FONT,
    )
    ribbon_toggle.pack(side=tk.LEFT, padx=2, pady=2)

    title_label = tk.Label(
        ribbon_header,
        text=title,
        bg=RIBBON_HEADER_BG,
        fg="white",
        font=(RIBBON_FONT[0], RIBBON_FONT[1], "bold"),
    )
    if not start_collapsed:
        title_label.pack(side=tk.LEFT, padx=4)

    ribbon_body = tk.Frame(ribbon_shell, bg=RIBBON_BG)
    if not start_collapsed:
        ribbon_body.pack(fill=tk.BOTH, expand=True)

    _canvas, ribbon_column = make_scrollable_column(ribbon_body)

    content_host = tk.Frame(outer_pane, bg="black")
    outer_pane.add(content_host, weight=1)

    def _measure_collapsed_width() -> int:
        parent.update_idletasks()
        w = int(ribbon_toggle.winfo_reqwidth()) + 4
        collapsed_width["px"] = max(w, 24)
        return collapsed_width["px"]

    def _collapse_ribbon():
        try:
            pos = int(outer_pane.sashpos(0))
            cw = collapsed_width["px"]
            if pos > cw + 10:
                last_width["px"] = max(MIN_EXPANDED_WIDTH, pos)
        except tk.TclError:
            pass
        title_label.pack_forget()
        ribbon_body.pack_forget()
        ribbon_toggle.config(text="▶")
        ribbon_open.set(False)
        try:
            outer_pane.pane(ribbon_shell, weight=0)
        except tk.TclError:
            pass
        try:
            outer_pane.sashpos(0, _measure_collapsed_width())
        except tk.TclError:
            pass

    def _expand_ribbon():
        title_label.pack(side=tk.LEFT, padx=4)
        ribbon_body.pack(fill=tk.BOTH, expand=True)
        ribbon_toggle.config(text="◀")
        ribbon_open.set(True)
        try:
            outer_pane.pane(ribbon_shell, weight=0)
        except tk.TclError:
            pass
        parent.update_idletasks()
        try:
            outer_pane.sashpos(0, last_width["px"])
        except tk.TclError:
            pass

    ribbon_toggle.config(command=lambda: _collapse_ribbon() if ribbon_open.get() else _expand_ribbon())

    def _remember_sash(_event=None):
        if ribbon_open.get():
            try:
                w = int(outer_pane.sashpos(0))
                cw = collapsed_width["px"]
                if w > cw + 10:
                    last_width["px"] = w
            except tk.TclError:
                pass
        action = _clamp_ribbon_sash(outer_pane, ribbon_open, collapsed_width)
        if action == "collapse":
            _collapse_ribbon()

    outer_pane.bind("<ButtonRelease-1>", _remember_sash, add="+")

    def _apply_startup_state():
        _measure_collapsed_width()
        try:
            if start_collapsed:
                _collapse_ribbon()
            else:
                outer_pane.sashpos(0, default_width)
        except tk.TclError:
            pass
        _clamp_ribbon_sash(outer_pane, ribbon_open, collapsed_width)

    try:
        outer_pane.sashpos(0, _measure_collapsed_width() if start_collapsed else default_width)
    except tk.TclError:
        pass
    parent.after_idle(_apply_startup_state)

    return {
        "outer_pane": outer_pane,
        "ribbon_shell": ribbon_shell,
        "ribbon_column": ribbon_column,
        "content_host": content_host,
        "collapse_ribbon": _collapse_ribbon,
        "expand_ribbon": _expand_ribbon,
    }

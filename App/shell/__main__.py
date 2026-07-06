"""Standalone demo: empty shell with placeholder panel frames.

Run::

    python -m App.shell
    python App/shell/__main__.py
"""

from __future__ import annotations

import tkinter as tk

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from App.shell.app_frame import DEFAULT_PANELS, build_shell_frame
    from App.shell.ribbon import collapsible_section, ribbon_button, ribbon_label
else:
    from .app_frame import DEFAULT_PANELS, build_shell_frame
    from .ribbon import collapsible_section, ribbon_button, ribbon_label


def _placeholder(host: tk.Frame, *, title: str, bg: str, fg: str, border: str) -> None:
    """Fill a panel host with an empty bordered frame and a label."""
    inner = tk.Frame(
        host,
        bg=bg,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
    )
    inner.pack(fill="both", expand=True, padx=12, pady=12)

    tk.Label(
        inner,
        text=title,
        bg=bg,
        fg=fg,
        font=("Segoe UI", 16, "bold"),
    ).pack(pady=(24, 8))

    tk.Label(
        inner,
        text="Empty host frame — mount widgets here.",
        bg=bg,
        fg=fg,
        font=("Segoe UI", 10),
        justify="center",
    ).pack(pady=(0, 24))


def _build_demo_ribbon(column: tk.Frame) -> None:
    ribbon_label(
        column,
        "Self-contained shell package.\nNo table, plots, or mesh loaded.",
        wraplength=200,
    )

    panels = collapsible_section(column, "Panels", expanded=True)
    ribbon_label(
        panels["body"],
        "Use ▾ on each panel title or right-click the grid.",
        wraplength=190,
    )

    actions = collapsible_section(column, "Demo", expanded=True)
    ribbon_button(actions["body"], "Print panel ids", command=lambda: print("panel_hosts", list(_PANEL_IDS)))


_PANEL_IDS: list[str] = []


def main() -> None:
    global _PANEL_IDS

    root = tk.Tk()
    root.title("Shell demo")
    root.geometry("960x720")
    root.minsize(640, 480)

    frame = build_shell_frame(root, ribbon_title="Shell demo", ribbon_width=220)
    _build_demo_ribbon(frame["ribbon_column"])

    specs = {spec["id"]: spec for spec in DEFAULT_PANELS}
    _PANEL_IDS = list(frame["panel_hosts"].keys())

    for panel_id, host in frame["panel_hosts"].items():
        spec = specs[panel_id]
        fg = "white" if spec["bg"] in ("#2a2a2a", "black") else "black"
        _placeholder(
            host,
            title=spec["title"],
            bg=spec["bg"],
            fg=fg,
            border=spec["border"],
        )

    root.mainloop()


if __name__ == "__main__":
    main()

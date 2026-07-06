"""Main window frame: ribbon split + dock grid with empty panel hosts.

This module only builds structure — no table, plots, or mesh widgets.
Pass the returned ``panel_hosts`` to application glue code.
"""

from __future__ import annotations

from typing import Callable, TypedDict

import tkinter as tk

from .dock_grid import DockGridLayout
from .ribbon import build_ribbon_shell


class PanelSpec(TypedDict):
    id: str
    title: str
    row: int
    col: int
    rowspan: int
    colspan: int
    border: str
    bg: str


DEFAULT_PANELS: tuple[PanelSpec, ...] = (
    {
        "id": "table",
        "title": "Table",
        "row": 0,
        "col": 0,
        "rowspan": 4,
        "colspan": 12,
        "border": "#4a6fa5",
        "bg": "#2a2a2a",
    },
    {
        "id": "plots",
        "title": "Signals",
        "row": 4,
        "col": 0,
        "rowspan": 8,
        "colspan": 7,
        "border": "#5a8a5a",
        "bg": "white",
    },
    {
        "id": "mesh",
        "title": "3D mesh",
        "row": 4,
        "col": 7,
        "rowspan": 8,
        "colspan": 5,
        "border": "#8a6a4a",
        "bg": "black",
    },
)


class ShellFrame(TypedDict):
    """Ribbon shell + dock grid. Widgets mount into ``panel_hosts``."""

    outer_pane: tk.Widget
    ribbon_shell: tk.Frame
    ribbon_column: tk.Frame
    content_host: tk.Frame
    collapse_ribbon: Callable[[], None]
    expand_ribbon: Callable[[], None]
    grid_main: tk.Frame
    dock_grid: DockGridLayout
    sections: dict[str, tk.Frame]
    panel_hosts: dict[str, tk.Frame]


def build_dock_layout(
    content_host: tk.Widget,
    *,
    panels: tuple[PanelSpec, ...] = DEFAULT_PANELS,
) -> dict:
    """Create the dock grid and empty content frames for each panel."""
    grid_main = tk.Frame(
        content_host,
        bg="#141414",
        highlightthickness=1,
        highlightbackground="#333333",
    )
    grid_main.pack(fill="both", expand=True)

    dock = DockGridLayout(grid_main)
    dock.pack(fill="both", expand=True)

    sections: dict[str, tk.Frame] = {}
    panel_hosts: dict[str, tk.Frame] = {}

    for spec in panels:
        panel_id = spec["id"]
        host = dock.create_host(
            panel_id,
            spec["title"],
            row=spec["row"],
            col=spec["col"],
            rowspan=spec["rowspan"],
            colspan=spec["colspan"],
        )
        section = tk.Frame(
            host,
            bg=spec["bg"],
            highlightthickness=2,
            highlightbackground=spec["border"],
        )
        section.pack(fill="both", expand=True, padx=1, pady=1)
        content = tk.Frame(section, bg=spec["bg"])
        content.pack(fill="both", expand=True, padx=2, pady=2)
        sections[panel_id] = section
        panel_hosts[panel_id] = content

    dock.focus_panel("table")

    return {
        "grid_main": grid_main,
        "dock_grid": dock,
        "sections": sections,
        "panel_hosts": panel_hosts,
    }


def build_shell_frame(
    parent: tk.Misc,
    *,
    ribbon_title: str = "Tools",
    ribbon_width: int = 240,
    panels: tuple[PanelSpec, ...] = DEFAULT_PANELS,
) -> ShellFrame:
    """Build ribbon split + dock grid. Returns empty panel hosts for glue to fill."""
    shell = build_ribbon_shell(
        parent,
        title=ribbon_title,
        default_width=ribbon_width,
    )
    dock_layout = build_dock_layout(shell["content_host"], panels=panels)
    return ShellFrame(
        outer_pane=shell["outer_pane"],
        ribbon_shell=shell["ribbon_shell"],
        ribbon_column=shell["ribbon_column"],
        content_host=shell["content_host"],
        collapse_ribbon=shell["collapse_ribbon"],
        expand_ribbon=shell["expand_ribbon"],
        grid_main=dock_layout["grid_main"],
        dock_grid=dock_layout["dock_grid"],
        sections=dock_layout["sections"],
        panel_hosts=dock_layout["panel_hosts"],
    )

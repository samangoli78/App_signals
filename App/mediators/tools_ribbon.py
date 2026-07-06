"""App-specific Tools sidebar — wired to :class:`~App.main_app.App` callbacks.

Generic ribbon chrome lives in :mod:`App.shell`; this module is glue only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

import tkinter as tk
from tkinter import ttk

from ..shell import (
    DockGridLayout,
    RibbonSection,
    collapsible_section,
    ribbon_button,
    ribbon_checkbox,
    ribbon_label,
)

if TYPE_CHECKING:
    from ..main_app import App


class LeftRibbonWidgets(TypedDict):
    """Widget handles assigned on ``App`` inside :meth:`~App.main_app.App.start`."""

    label: tk.Label
    button_dropdown: tk.Button
    check_boxes: dict[str, tk.IntVar]
    show_original_labels_var: tk.IntVar
    button_trip: tk.Button
    button_VT: tk.Button
    button_compute_all: tk.Button
    button_compute_cv: tk.Button
    pred_label: tk.Label
    ml_model_var: tk.StringVar
    ml_combo: ttk.Combobox
    panel_vis_vars: NotRequired[dict[str, tk.IntVar]]


def _build_panel_ribbon_section(ribbon_column: tk.Frame, dock: DockGridLayout) -> dict[str, tk.IntVar]:
    panels: RibbonSection = collapsible_section(ribbon_column, "Panels")
    panel_vis_vars: dict[str, tk.IntVar] = {}
    for panel_id, title, vis_var in dock.panel_items():
        panel_vis_vars[panel_id] = vis_var
        ribbon_checkbox(
            panels["body"],
            title,
            vis_var,
            lambda pid=panel_id: dock.on_panel_vis_toggle(pid),
        )
    ribbon_button(
        panels["body"],
        "Panel menu…",
        lambda anchor=panels["body"]: dock.popup_panel_menu(anchor=anchor),
    )
    ribbon_button(panels["body"], "Show all panels", dock.show_all_panels)
    ribbon_button(panels["body"], "Hide all panels", dock.hide_all_panels)
    return panel_vis_vars


def build_left_ribbon(
    app: App,
    ribbon_column: tk.Frame,
    dock: DockGridLayout | None = None,
) -> LeftRibbonWidgets:
    """Populate the shell ribbon column with navigation, filters, ML, etc."""
    widgets: LeftRibbonWidgets = {}  # type: ignore[typeddict-item]

    nav: RibbonSection = collapsible_section(ribbon_column, "Navigation")
    widgets["label"] = ribbon_label(
        nav["body"],
        f"point {app.cont[app.i][0]['point number'].values[app.j]}",
    )
    widgets["button_dropdown"] = ribbon_button(nav["body"], "Options", app.drop_down)

    filters: RibbonSection = collapsible_section(ribbon_column, "View filters")
    check_boxes: dict[str, tk.IntVar] = {"Energy": None, "Only_Green": None}  # type: ignore[misc]
    for key in check_boxes:
        check_boxes[key] = tk.IntVar()
        ribbon_checkbox(filters["body"], key, check_boxes[key], app.checker)
    widgets["check_boxes"] = check_boxes

    show_original_labels_var = tk.IntVar(
        value=1 if getattr(app, "show_original_labels", False) else 0,
    )
    ribbon_checkbox(
        filters["body"],
        "Original labels",
        show_original_labels_var,
        app._toggle_original_labels,
    )
    widgets["show_original_labels_var"] = show_original_labels_var

    if dock is not None:
        widgets["panel_vis_vars"] = _build_panel_ribbon_section(ribbon_column, dock)

    protocols: RibbonSection = collapsible_section(ribbon_column, "Protocols")
    widgets["button_trip"] = ribbon_button(
        protocols["body"],
        "Switch to Triple Extra Protocol",
        app.triple_protocol,
    )
    widgets["button_VT"] = ribbon_button(
        protocols["body"],
        "Switch to VT Protocol",
        app.VT_protocol,
    )

    compute: RibbonSection = collapsible_section(ribbon_column, "Compute")
    ribbon_button(
        compute["body"],
        "Screenshot",
        lambda name=None: app.capture_window(name),
    )
    widgets["button_compute_all"] = ribbon_button(
        compute["body"],
        "Compute all",
        app._compute_all_clicked,
    )
    widgets["button_compute_cv"] = ribbon_button(
        compute["body"],
        "Compute Conduction Velocity",
        app._compute_conduction_velocity_clicked,
    )

    ml: RibbonSection = collapsible_section(ribbon_column, "ML model")
    ribbon_label(ml["body"], "Model:")
    ml_model_var = tk.StringVar(value="(none)")
    ml_combo = ttk.Combobox(
        ml["body"],
        textvariable=ml_model_var,
        values=["(none)"],
        state="readonly",
    )
    ml_combo.pack(fill="x", padx=6, pady=2)
    ml_combo.bind("<<ComboboxSelected>>", app._on_ml_model_change)
    ribbon_button(ml["body"], "Load model…", app._open_ml_dialog)
    widgets["pred_label"] = ribbon_label(
        ml["body"],
        "Predicted: —",
        bold=True,
        wraplength=200,
    )
    widgets["ml_model_var"] = ml_model_var
    widgets["ml_combo"] = ml_combo

    return widgets

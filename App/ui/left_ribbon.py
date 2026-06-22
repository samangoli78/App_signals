"""Collapsible left ribbon for the main application window."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .ribbon_widgets import (
    build_ribbon_shell,
    collapsible_section,
    ribbon_button,
    ribbon_checkbox,
    ribbon_label,
)


def build_app_shell(parent):
    return build_ribbon_shell(parent, title="Tools", default_width=240)


def build_left_ribbon(app, ribbon_column):
    """Build all toolbar controls inside collapsible ribbon sections."""
    widgets = {}

    nav = collapsible_section(ribbon_column, "Navigation")
    widgets["label"] = ribbon_label(
        nav["body"],
        f"point {app.cont[app.i][0]['point number'].values[app.j]}",
    )
    widgets["button_dropdown"] = ribbon_button(nav["body"], "Options", app.drop_down)

    filters = collapsible_section(ribbon_column, "View filters")
    check_boxes = {"Energy": None, "Only_Green": None}
    for key in check_boxes:
        check_boxes[key] = tk.IntVar()
        ribbon_checkbox(filters["body"], key, check_boxes[key], app.checker)
    widgets["check_boxes"] = check_boxes
    show_original_labels_var = tk.IntVar(
        value=1 if getattr(app, "show_original_labels", False) else 0
    )
    ribbon_checkbox(
        filters["body"],
        "Original labels",
        show_original_labels_var,
        app._toggle_original_labels,
    )
    widgets["show_original_labels_var"] = show_original_labels_var

    protocols = collapsible_section(ribbon_column, "Protocols")
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

    compute = collapsible_section(ribbon_column, "Compute")
    ribbon_button(compute["body"], "Screenshot", lambda name=None: app.capture_window(name))
    widgets["button_compute_all"] = ribbon_button(
        compute["body"], "Compute all", app._compute_all_clicked
    )
    widgets["button_global_patch"] = ribbon_button(
        compute["body"], "Compute patch dv/ds", app._compute_global_patch_clicked
    )
    widgets["button_compute_cv"] = ribbon_button(
        compute["body"],
        "Compute Conduction Velocity",
        app._compute_conduction_velocity_clicked,
    )

    ml = collapsible_section(ribbon_column, "ML model")
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
        ml["body"], "Predicted: —", bold=True, wraplength=200
    )
    widgets["ml_model_var"] = ml_model_var
    widgets["ml_combo"] = ml_combo
    return widgets

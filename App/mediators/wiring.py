"""Default glue classes used when assembling the main app."""

from __future__ import annotations

from dataclasses import dataclass

from ..plotting import PlotPresenter
from .app_glue import AppLayoutGlue, MeshAppGlue, TableAppGlue


@dataclass(frozen=True)
class AppWiring:
    presenter_cls: type = PlotPresenter
    table_glue_cls: type = TableAppGlue
    mesh_glue_cls: type = MeshAppGlue
    layout_glue_cls: type = AppLayoutGlue


DEFAULT_WIRING = AppWiring()

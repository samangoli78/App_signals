"""Mediators bridge the main App to feature packages (table, plots, 3D viewer)."""
from .app_glue import AppLayoutGlue, MeshAppGlue, TableAppGlue

# Backward-compatible alias
ContentLayoutGlue = AppLayoutGlue

__all__ = ["TableAppGlue", "MeshAppGlue", "AppLayoutGlue", "ContentLayoutGlue"]

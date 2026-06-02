"""UI structure: matplotlib overlays and Tk dialogs.

Pulled into one place so ``main_app`` only knows about the package surface:

    from App.ui import Area, Shades, Toplevel
"""
from .shades import Shades
from .area import Area
from .toplevel import Toplevel

__all__ = ["Area", "Shades", "Toplevel"]

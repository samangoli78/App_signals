"""Standalone demo: 3D mesh panel (loads carto study via folder dialog).

Run: python -m App.viewer3d
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

from ..carto import Carto
from .ui import CartoMeshPanel


def main() -> None:
    root = tk.Tk()
    root.title("3D viewer demo")
    root.geometry("900x700")

    try:
        carto = Carto()
    except Exception as exc:
        messagebox.showerror("Carto", f"Could not load study:\n{exc}")
        root.destroy()
        return

    panel = CartoMeshPanel(root, carto)
    panel.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()

"""Standalone demo: styled signal figure with synthetic traces.

Run: python -m App.plotting
"""

from __future__ import annotations

import tkinter as tk

import numpy as np

from .figure_host import SignalFigureHost


def main() -> None:
    root = tk.Tk()
    root.title("Plot style demo")
    root.geometry("800x500")

    host_frame = tk.Frame(root, bg="white")
    host_frame.pack(fill="both", expand=True)
    host = SignalFigureHost(host_frame)

    t = np.linspace(0, 2.5, 500)
    host.axes["top"].plot(t, 0.3 * np.sin(2 * np.pi * 2 * t), color="C0")
    host.axes["mid"].plot(t, 0.5 * np.sin(2 * np.pi * 2 * t + 0.4), color="C1")
    host.axes["bot"].plot(t, 2.0 * np.sin(2 * np.pi * 2 * t), color="C2")
    for ax in host.axes.values():
        ax.set_xlabel("Time (s)")
    host.canvas.draw()

    root.mainloop()


if __name__ == "__main__":
    main()

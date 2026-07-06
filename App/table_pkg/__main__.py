"""Standalone demo: editable table on sample data.

Run: python -m App.table_pkg
"""

from __future__ import annotations

import tkinter as tk

import pandas as pd

from .widget import TableWidget


def main() -> None:
    root = tk.Tk()
    root.title("Table demo")
    root.geometry("640x400")

    df = pd.DataFrame(
        {
            "point number": [101, 102, 103, 104],
            "label_color": ["Green", "Red", "Green", "Blue"],
            "delta": ["1.2", "0.8", "—", "2.1"],
        }
    )
    table = TableWidget(root, df)
    table.pack(fill="both", expand=True)

    def on_select(ctx):
        print("select", ctx)

    def on_edit(ctx):
        print("edit", ctx)

    table.tree.on_select_row = on_select
    table.tree.on_edit_commit = on_edit

    root.mainloop()


if __name__ == "__main__":
    main()

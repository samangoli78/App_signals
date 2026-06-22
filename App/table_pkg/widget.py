import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np


class EditableTree(ttk.Treeview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.defaults: dict[int, list[str]] = {}
        self.on_cell_move = lambda ctx: None
        self.on_edit_commit = lambda ctx: None
        self.on_select_row = lambda ctx: None

        self.cur_iid: str | None = None
        self.cur_col: int = 0
        self._edit_entry: tk.Entry | None = None

        self.tag_configure("active_row", background="#E8F0FE")
        self.tag_configure("changed", background="#A5A5A5")

        self.bind("<Button-1>", self._ev_click_select, add="+")
        self.bind("<Double-Button-1>", self._ev_double_click, add="+")
        self.bind("<Button-3>", self._ev_popup, add="+")
        self.bind("<Up>", self._ev_nav_ud, add="+")
        self.bind("<Down>", self._ev_nav_ud, add="+")
        self.bind("<Left>", self._ev_nav_lr, add="+")
        self.bind("<Right>", self._ev_nav_lr, add="+")
        self.bind("<Return>", self._ev_enter, add="+")
        self.bind("<Escape>", self._ev_escape, add="+")

    def init_from_df(self, df, *, keep_col_widths=True, reset_cur_col=True):
        df = df.reset_index()
        self.df = df
        self.close_editor()
        for iid in self.get_children():
            self.delete(iid)
        self.cur_iid = None
        if reset_cur_col:
            self.cur_col = 0

        new_cols = [str(c) for c in df.columns]
        old_cols = list(self["columns"]) if self["columns"] else []
        widths = {}
        if keep_col_widths and old_cols:
            for c in old_cols:
                try:
                    widths[c] = self.column(c, "width")
                except Exception:
                    pass

        self["columns"] = new_cols
        for c in new_cols:
            self.heading(c, text=c)
            self.column(c, width=widths.get(c, 120), minwidth=50, stretch=True, anchor="center")

        for i, (_, row) in enumerate(df.iterrows()):
            iid = f"row{i}"
            self.insert("", "end", iid=iid, values=[row[c] for c in df.columns])

        kids = self.get_children()
        if kids:
            self.cur_iid = kids[0]
            self.selection_set(self.cur_iid)
            self.focus(self.cur_iid)
            self.see(self.cur_iid)
            self._set_active_row_tag()
        else:
            self.selection_remove(self.selection())
            self.focus("")

    def update_column_values(self, name, values) -> None:
        """Update one column in place without rebuilding rows or selection."""
        name = str(name)
        cols = list(self["columns"])
        if name not in cols:
            return
        ci = cols.index(name)
        for ri, iid in enumerate(self.get_children()):
            if ri >= len(values):
                break
            vals = self._get_values(iid)
            while len(vals) <= ci:
                vals.append("")
            vals[ci] = values[ri]
            self.item(iid, values=vals)
        if hasattr(self, "df") and self.df is not None and name in self.df.columns:
            n = min(len(values), len(self.df))
            self.df.loc[: n - 1, name] = list(values)[:n]

    def insert_column(self, name, values, *, after=None, width=120) -> None:
        """Insert a column without rebuilding the tree."""
        name = str(name)
        cols = list(self["columns"])
        if name in cols:
            self.update_column_values(name, values)
            return

        if after is not None and str(after) in cols:
            idx = cols.index(str(after)) + 1
        else:
            idx = len(cols)

        new_cols = cols[:idx] + [name] + cols[idx:]
        new_defaults: dict[int, list[str]] = {}
        for col_idx, options in self.defaults.items():
            new_key = col_idx + 1 if col_idx >= idx else col_idx
            new_defaults[new_key] = options
        self.defaults = new_defaults

        self["columns"] = new_cols
        self.heading(name, text=name)
        self.column(name, width=width, minwidth=50, stretch=True, anchor="center")

        for ri, iid in enumerate(self.get_children()):
            vals = self._get_values(iid)
            val = values[ri] if ri < len(values) else ""
            vals.insert(idx, val)
            self.item(iid, values=vals)

        if self.cur_col >= idx:
            self.cur_col += 1

    def remove_column(self, name) -> None:
        """Remove a column without rebuilding the tree."""
        name = str(name)
        cols = list(self["columns"])
        if name not in cols:
            return
        idx = cols.index(name)

        new_defaults: dict[int, list[str]] = {}
        for col_idx, options in self.defaults.items():
            if col_idx == idx:
                continue
            new_key = col_idx - 1 if col_idx > idx else col_idx
            new_defaults[new_key] = options
        self.defaults = new_defaults

        self["columns"] = [c for c in cols if c != name]
        for iid in self.get_children():
            vals = self._get_values(iid)
            if idx < len(vals):
                del vals[idx]
            self.item(iid, values=vals)

        if self.cur_col == idx:
            self.cur_col = max(0, idx - 1)
        elif self.cur_col > idx:
            self.cur_col -= 1

        if hasattr(self, "df") and self.df is not None and name in self.df.columns:
            self.df = self.df.drop(columns=[name])

    def _ensure_current_cell(self):
        kids = self.get_children()
        if not kids:
            self.cur_iid = None
            return
        if self.cur_iid in kids:
            return
        sel = self.selection()
        if sel and sel[0] in kids:
            self.cur_iid = sel[0]
            return
        self.cur_iid = kids[0]
        self.selection_set(self.cur_iid)
        self.focus(self.cur_iid)

    def _row_index_from_iid(self, iid: str) -> int | None:
        try:
            return self.get_children().index(iid)
        except ValueError:
            return None

    def _col_count(self) -> int:
        return len(self["columns"])

    def _get_values(self, iid: str) -> list:
        return list(self.item(iid).get("values") or [])

    def _bbox_of(self, iid: str, col: int):
        if iid is None or col is None:
            return None
        if col < 0 or col >= self._col_count():
            return None
        bbox = self.bbox(iid, self["columns"][col])
        return bbox if bbox else None

    def _cell_ctx(self, key: str):
        self._ensure_current_cell()
        iid = self.cur_iid
        col = self.cur_col
        row = self._row_index_from_iid(iid) if iid else None
        values = self._get_values(iid) if iid else None
        cell_value = None
        if values is not None and 0 <= col < len(values):
            cell_value = values[col]
        return {"key": key, "iid": iid, "row": row, "col": col, "values": values, "cell_value": cell_value}

    def _set_active_row_tag(self):
        for iid in self.get_children():
            tags = self.item(iid).get("tags") or ()
            self.item(iid, tags=tuple(t for t in tags if t != "active_row"))
        if self.cur_iid:
            tags = set(self.item(self.cur_iid).get("tags") or ())
            tags.add("active_row")
            self.item(self.cur_iid, tags=tuple(tags))

    def close_editor(self):
        if self._edit_entry is not None:
            try:
                self._edit_entry.destroy()
            except Exception:
                pass
            self._edit_entry = None

    def edit_current_cell(self):
        self._ensure_current_cell()
        iid, col = self.cur_iid, self.cur_col
        bbox = self._bbox_of(iid, col)
        if not bbox:
            return
        self.close_editor()
        x, y, w, h = bbox
        vals = self._get_values(iid)
        if col < 0 or col >= len(vals):
            return
        entry = tk.Entry(self.master)
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, vals[col])
        entry.select_range(0, tk.END)
        entry.focus_set()
        entry.bind("<FocusOut>", lambda e: self.close_editor(), add="+")
        entry.bind("<Escape>", lambda e: (self.close_editor(), self.focus_force()), add="+")
        entry.bind("<Up>", lambda event: (self.close_editor(), self.focus_force()), add="+")
        entry.bind("<Down>", lambda event: (self.close_editor(), self.focus_force()), add="+")
        self._edit_entry = entry

        def go_to_next():
            n = int(self.cur_iid.split("row")[-1]) + 1
            next_iid = f"row{n}"
            self.cur_iid = next_iid
            try:
                self.selection_set(next_iid)
                self.focus(next_iid)
                self.see(next_iid)
            except Exception:
                self._ensure_current_cell()

        def commit(value: str):
            vals2 = self._get_values(iid)
            if 0 <= col < len(vals2):
                vals2[col] = value
                tags = set(self.item(iid).get("tags") or ())
                tags.add("changed")
                self.item(iid, values=vals2, tags=tuple(tags))
                self.on_edit_commit({"iid": iid, "row": self._row_index_from_iid(iid), "col": col, "value": value, "values": vals2})

        def on_entry_enter(e):
            commit(entry.get())
            self.close_editor()
            self.focus_force()
            go_to_next()
            self.edit_current_cell()

        if col in self.defaults and self.defaults[col]:
            menu = tk.Menu(self.master, tearoff=False)

            def close_menu(event=None):
                try:
                    menu.unpost()
                except Exception:
                    pass
                try:
                    menu.destroy()
                except Exception:
                    pass
                self.focus_force()
                self.close_editor()

            def pick(item):
                commit(item)
                close_menu()
                go_to_next()
                self.edit_current_cell()

            for item in self.defaults[col]:
                menu.add_command(label=item, command=lambda it=item: pick(it))
            menu.bind("<Escape>", close_menu)
            sx = self.winfo_rootx() + x
            sy = self.winfo_rooty() + y + h
            menu.tk_popup(sx, sy)

        entry.bind("<Return>", on_entry_enter, add="+")

    def _ev_escape(self, event):
        self.close_editor()
        self.focus_force()
        return None

    def _ev_click_select(self, event):
        def after():
            sel = self.selection()
            if sel:
                self.cur_iid = sel[0]
            self._set_active_row_tag()
            self.on_select_row(self._cell_ctx("Click"))
        self.after_idle(after)

    def _ev_double_click(self, event):
        if self.identify_region(event.x, event.y) == "cell":
            iid = self.identify_row(event.y)
            col_str = self.identify_column(event.x)
            if iid and col_str.startswith("#"):
                self.cur_iid = iid
                self.cur_col = int(col_str[1:]) - 1
                self.selection_set(iid)
                self.focus(iid)
                self._set_active_row_tag()
                self.edit_current_cell()
        return None

    def _ev_nav_ud(self, event):
        if event.widget.focus_get() is not event.widget:
            return None
        def after():
            sel = self.selection()
            if sel:
                self.cur_iid = sel[0]
            self._set_active_row_tag()
            self.on_cell_move(self._cell_ctx(event.keysym))
        self.after_idle(after)
        return None

    def _ev_nav_lr(self, event):
        if event.widget.focus_get() is not event.widget:
            return None
        self._ensure_current_cell()
        if event.keysym == "Left":
            self.cur_col = max(0, self.cur_col - 1)
        else:
            self.cur_col = min(self._col_count() - 1, self.cur_col + 1)
        self.on_cell_move(self._cell_ctx(event.keysym))
        return "break"

    def _ev_enter(self, event):
        if event.widget.focus_get() is not event.widget:
            return None
        self.close_editor()
        self.on_cell_move(self._cell_ctx("Return"))
        self.edit_current_cell()
        return "break"

    def _ev_popup(self, event):
        iid = self.identify_row(event.y)
        if iid:
            self.selection_set(iid)
            self.focus(iid)
            self.cur_iid = iid
            self._set_active_row_tag()
        menu = tk.Menu(self.master, tearoff=False)

        def close_menu(event=None):
            try:
                menu.unpost()
            except Exception:
                pass
            try:
                menu.destroy()
            except Exception:
                pass
            self.focus_force()

        def edit_here():
            if self.identify_region(event.x, event.y) == "cell":
                col_str = self.identify_column(event.x)
                if col_str.startswith("#"):
                    self.cur_col = int(col_str[1:]) - 1
            self.edit_current_cell()
            close_menu()

        menu.add_command(label="Edit cell", command=edit_here)
        menu.add_command(label="Save", command=self.save_to_file)
        menu.add_command(label="load", command=self.load_csv)
        menu.add_separator()
        menu.add_command(label="Close", command=close_menu)
        menu.bind("<Escape>", close_menu)
        menu.tk_popup(event.x_root, event.y_root)

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Load table data",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not load file:\n{path}\n\n{e}")
            return
        df.columns = [str(c) for c in df.columns]
        df = df.fillna("")
        self.init_from_df(df)

    def save_to_file(self):
        file = filedialog.asksaveasfilename(
            confirmoverwrite=True,
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"), ("Text", "*.txt"), ("All Files", "*.*")),
        )
        if not file:
            return
        cols = list(self["columns"])
        rows = [self._get_values(iid) for iid in self.get_children()]
        ext = file.split(".")[-1].lower()
        if ext == "txt":
            with open(file, "w", encoding="utf-8") as f:
                f.write(";".join(map(str, cols)) + "\n")
                for r in rows:
                    f.write(";".join(map(str, r)) + "\n")
        else:
            pd.DataFrame(rows, columns=cols).to_csv(file, index=False)


class TableWidget(tk.Frame):
    def __init__(self, master, df: pd.DataFrame):
        super().__init__(master)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.tree = EditableTree(self, columns=(), show="headings", selectmode="browse")
        vs = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hs = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        self.tree.init_from_df(df)

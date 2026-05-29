import tkinter as tk
from tkinter import ttk, filedialog,messagebox
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

        # row highlight tags
        self.tag_configure("active_row", background="#E8F0FE")
        self.tag_configure("changed", background="#A5A5A5")

        # bindings
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
        """
        Data-layer re-init (index is always dropped):
        - clears rows
        - uses df.columns only
        - inserts df rows
        - resets cursor/selection to a known start state
        """

        # --- normalize dataframe ---
        # drop index semantics entirely
        df = df.reset_index()
        self.df = df

        # --- close active editor ---
        self.close_editor()

        # --- clear rows ---
        for iid in self.get_children():
            self.delete(iid)

        # --- reset cursor state ---
        self.cur_iid = None
        if reset_cur_col:
            self.cur_col = 0

        # --- set table schema ---
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
            self.column(
                c,
                width=widths.get(c, 120),
                minwidth=50,
                stretch=True,
                anchor="center",
            )

        # --- insert rows ---
        for i, (_, row) in enumerate(df.iterrows()):
            iid = f"row{i}"
            self.insert(
                "",
                "end",
                iid=iid,
                values=[row[c] for c in df.columns],
            )

        # --- reset selection/focus ---
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


    # ---------------------------
    # Helpers
    # ---------------------------

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
        return {
            "key": key,
            "iid": iid,
            "row": row,
            "col": col,
            "values": values,
            "cell_value": cell_value,
        }

    def _set_active_row_tag(self):
        # clear tag from all rows (cheap enough for moderate sizes)
        for iid in self.get_children():
            tags = self.item(iid).get("tags") or ()
            new_tags = tuple(t for t in tags if t != "active_row")
            self.item(iid, tags=new_tags)

        if self.cur_iid:
            tags = set(self.item(self.cur_iid).get("tags") or ())
            tags.add("active_row")
            self.item(self.cur_iid, tags=tuple(tags))

    # ---------------------------
    # Editor lifecycle
    # ---------------------------

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

        # destroy editor on focus out
        entry.bind("<FocusOut>", lambda e: self.close_editor(), add="+")
        entry.bind("<Escape>", lambda e: (self.close_editor(), self.focus_force()), add="+")
        entry.bind("<Up>",   lambda event: (self.close_editor(), self.focus_force()), add="+")
        entry.bind("<Down>",lambda event: (self.close_editor(), self.focus_force()), add="+")

        self._edit_entry = entry
        def go_to_next():
            n = int(self.cur_iid.split("row")[-1]) + 1
            next_iid=f"row{n}"
            self.cur_iid = next_iid
            try:
                self.selection_set(next_iid)
                self.focus(next_iid)
                self.see(next_iid)
            except Exception as e:
                print(e)
                self._ensure_current_cell()

        def on_entry_Enter(e):
            commit(entry.get())
            self.close_editor()
            self.focus_force()
            go_to_next()
            self.edit_current_cell()

        def commit(value: str):
            vals2 = self._get_values(iid)
            if 0 <= col < len(vals2):
                vals2[col] = value
                # keep any existing tags, add "changed"
                tags = set(self.item(iid).get("tags") or ())
                tags.add("changed")
                self.item(iid, values=vals2, tags=tuple(tags))
                self.on_edit_commit({
                    "iid": iid,
                    "row": self._row_index_from_iid(iid),
                    "col": col,
                    "value": value,
                    "values": vals2,
                })

        # dropdown defaults (optional)
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
                self.close_editor()  # cancel edit when leaving dropdown

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
        
        entry.bind("<Return>", on_entry_Enter , add="+")
        

    # ---------------------------
    # Events
    # ---------------------------

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
            ctx = self._cell_ctx(event.keysym)
            self.on_cell_move(ctx)

        self.after_idle(after)
        return None  # keep Treeview native Up/Down

    def _ev_nav_lr(self, event):
        if event.widget.focus_get() is not event.widget:
            return None

        
        self._ensure_current_cell()

        if event.keysym == "Left":
            self.cur_col = max(0, self.cur_col - 1)
        else:
            self.cur_col = min(self._col_count() - 1, self.cur_col + 1)

        ctx = self._cell_ctx(event.keysym)
        self.on_cell_move(ctx)
        return "break"  # stop horizontal scrolling

    def _ev_enter(self, event):
        if event.widget.focus_get() is not event.widget:
            return None
        self.close_editor()
        ctx = self._cell_ctx("Return")
        self.on_cell_move(ctx)
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

    # ---------------------------
    # Save / export
    # ---------------------------
    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Load table data",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return  # user cancelled

        try:
            # auto-detect separator; works for csv and most txt
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
            )
        except Exception as e:
            messagebox.showerror(
                "Load failed",
                f"Could not load file:\n{path}\n\n{e}",
            )
            return

        # normalize column names to strings (Treeview requires str)
        df.columns = [str(c) for c in df.columns]

        # optional: replace NaN with empty string for editing
        df = df.fillna("")

        # data-layer re-init (your function)
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



if __name__ == "__main__":
    root = tk.Tk()
    root.title("EditableTree demo (visible table)")
    root.geometry("900x380")

    df = pd.DataFrame(
        np.random.randint(0, 100, (12, 6)),
        columns=["A", "B", "C", "D", "E", "F"],
    )

    table = TableWidget(root, df)
    table.pack(fill="both", expand=True, padx=8, pady=8)

    tv = table.tree
    tv.defaults = {3: ["LOW", "MEDIUM", "HIGH"]}  # col=3 is "C" (index,A,B,C)

    def on_move(ctx):
        print(f"[MOVE] {ctx['key']} row={ctx['row']} col={ctx['col']} value={ctx['cell_value']}")

    def on_commit(ctx):
        print(f"[COMMIT] row={ctx['row']} col={ctx['col']} value={ctx['value']}")

    def on_select(ctx):
        print(f"[SELECT] row={ctx['row']} col={ctx['col']}")

    tv.on_cell_move = on_move
    tv.on_edit_commit = on_commit
    tv.on_select_row = on_select

    # initial state
    tv.focus_set()
    kids = tv.get_children()
    if kids:
        tv.selection_set(kids[0])
        tv.focus(kids[0])
        tv.cur_iid = kids[0]
        tv.cur_col = 1  # start on column "A"
        tv._set_active_row_tag()

    root.mainloop()

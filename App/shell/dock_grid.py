"""Dock layout: preview rectangle on canvas, panels placed with matching pixel rects.

Panels are ``Frame`` children of the grid. Drag the title bar to move; drag edges to resize.
"""

from __future__ import annotations

from typing import Callable

import tkinter as tk


class DockGridLayout(tk.Frame):
    """Fixed rows x cols. Drag/resize shows clamped preview; release places panel there."""

    _EDGE_PX = 6
    _CORNER_PX = 8
    _DRAG_THRESHOLD_PX = 5
    _RESIZE_EDGES = (
        ("n", "size_ns", 0.0, 0.0, "n"),
        ("s", "size_ns", 0.0, 1.0, "s"),
        ("e", "size_we", 1.0, 0.0, "e"),
        ("w", "size_we", 0.0, 0.0, "w"),
    )
    _RESIZE_CORNERS = (
        ("nw", "top_left_corner", 0.0, 0.0, "nw"),
        ("ne", "top_right_corner", 1.0, 0.0, "ne"),
        ("sw", "bottom_left_corner", 0.0, 1.0, "sw"),
        ("se", "bottom_right_corner", 1.0, 1.0, "se"),
    )

    def __init__(self, master, *, rows: int = 12, cols: int = 12, **kw) -> None:
        super().__init__(master, **kw)
        self._rows = int(rows)
        self._cols = int(cols)
        self._panels: dict[str, dict] = {}
        self._op_id: str | None = None
        self._op_kind: str | None = None
        self._resize_mode: str | None = None
        self._resize_anchor: tuple[int, int, int, int] | None = None
        self._preview_placement: tuple[int, int, int, int] | None = None
        self._last_grid_size: tuple[int, int] | None = None
        self._on_layout_changed: Callable[[str], None] | None = None
        self._window_layout_after: str | None = None
        self._z_order: list[str] = []
        self._title_press: tuple[int, int, str] | None = None

        self._holder = tk.Frame(self)
        self._holder.pack(fill="both", expand=True)

        self._grid = tk.Frame(self._holder, bg="#1e1e1e")
        self._grid.pack(fill="both", expand=True)

        self._grid_canvas = tk.Canvas(self._grid, highlightthickness=0, bd=0, bg="#1e1e1e")
        self._grid_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._grid.bind("<Configure>", self._on_grid_configure)
        self._grid.bind("<Button-3>", self._show_panel_menu)
        self._grid_canvas.bind("<Button-3>", self._show_panel_menu)
        self.bind_all("<B1-Motion>", self._motion, add="+")
        self.bind_all("<ButtonRelease-1>", self._release, add="+")
        self.bind_all("<ButtonPress-1>", self._on_global_press, add="+")

    # ------------------------------------------------------------------ public

    def create_host(
        self,
        panel_id: str,
        title: str,
        *,
        row: int,
        col: int,
        rowspan: int,
        colspan: int,
        visible: bool = True,
    ) -> tk.Frame:
        shell = tk.Frame(self._grid, bd=0, bg="#2a2a2a")

        bar = tk.Frame(shell, bg="#525252")
        bar.pack(fill="x")

        title_lbl = tk.Label(
            bar,
            text=f"  {title}",
            bg="#525252",
            fg="white",
            anchor="w",
            cursor="fleur",
        )
        title_lbl.pack(side="left", fill="x", expand=True)
        menu_btn = tk.Label(bar, text=" ▾ ", bg="#525252", fg="#cccccc", cursor="hand2")
        menu_btn.pack(side="right")

        host = tk.Frame(shell, bg=shell["bg"])
        host.pack(fill="both", expand=True)

        placement = self._clamp(row, col, rowspan, colspan)
        vis_var = tk.IntVar(value=1 if visible else 0)
        self._panels[panel_id] = {
            "title": title,
            "shell": shell,
            "host": host,
            "placement": placement,
            "visible": bool(visible),
            "vis_var": vis_var,
        }

        title_lbl.bind("<ButtonPress-1>", lambda e, pid=panel_id: self._on_title_press(e, pid))
        title_lbl.bind("<B1-Motion>", lambda e, pid=panel_id: self._on_title_motion(e, pid))
        title_lbl.bind("<ButtonRelease-1>", lambda e, pid=panel_id: self._on_title_release(e, pid))
        menu_btn.bind("<Button-1>", self._show_panel_menu)
        self._add_grips(shell, panel_id)
        if panel_id not in self._z_order:
            self._z_order.append(panel_id)

        if visible:
            self._place_panel(panel_id)
        else:
            shell.place_forget()
        self._draw_grid_lines()
        return host

    def set_layout_change_callback(self, callback: Callable[[str], None] | None) -> None:
        self._on_layout_changed = callback

    def focus_panel(self, panel_id: str) -> None:
        p = self._panels.get(panel_id)
        if p is None or not p["visible"]:
            return
        if self._op_id is not None and panel_id != self._op_id:
            return
        if panel_id in self._z_order:
            self._z_order.remove(panel_id)
        self._z_order.append(panel_id)
        try:
            p["shell"].tkraise()
        except tk.TclError:
            pass
        self._sync_stack()

    def _widget_in_grid(self, widget) -> bool:
        if not isinstance(widget, tk.Misc):
            return False
        while widget is not None:
            if widget == self._grid:
                return True
            widget = widget.master
        return False

    def _panel_for_widget(self, widget) -> str | None:
        if not isinstance(widget, tk.Misc):
            return None
        while widget is not None:
            for pid, p in self._panels.items():
                if widget == p["shell"]:
                    return pid
            widget = widget.master
        return None

    def _panel_at_xy(self, x_root: int, y_root: int) -> str | None:
        hits: list[str] = []
        for pid, p in self._panels.items():
            if not p["visible"]:
                continue
            x0, y0, x1, y1 = self._rect_px(p["placement"])
            gx, gy = self._grid.winfo_rootx(), self._grid.winfo_rooty()
            if gx + x0 <= x_root < gx + x1 and gy + y0 <= y_root < gy + y1:
                hits.append(pid)
        if not hits:
            return None
        for pid in reversed(self._z_order):
            if pid in hits:
                return pid
        return hits[-1]

    def _on_global_press(self, event) -> None:
        if self._op_id is not None:
            return
        pid = self._panel_for_widget(event.widget)
        if pid is not None:
            self.focus_panel(pid)
            return
        if not self._widget_in_grid(event.widget):
            return
        pid = self._panel_at_xy(event.x_root, event.y_root)
        if pid is not None:
            self.focus_panel(pid)

    def _sync_stack(self) -> None:
        try:
            self._grid.tk.call("lower", self._grid_canvas._w)
        except tk.TclError:
            pass
        for pid in self._z_order:
            p = self._panels.get(pid)
            if p is not None and p["visible"]:
                try:
                    p["shell"].tkraise()
                except tk.TclError:
                    pass

    def set_visible(self, panel_id: str, visible: bool) -> None:
        p = self._panels.get(panel_id)
        if p is None:
            return
        p["visible"] = bool(visible)
        p["vis_var"].set(1 if visible else 0)
        if visible:
            p.pop("_last_place", None)
            self._place_panel(panel_id, raise_it=True)
            self.focus_panel(panel_id)
            self._grid.update_idletasks()
            self._notify_layout(panel_id)
        else:
            p["shell"].place_forget()
            p.pop("_last_place", None)
        self._clear_preview()
        self._draw_grid_lines()

    def panel_items(self) -> list[tuple[str, str, tk.IntVar]]:
        return [(pid, p["title"], p["vis_var"]) for pid, p in self._panels.items()]

    def on_panel_vis_toggle(self, panel_id: str) -> None:
        self.set_visible(panel_id, bool(self._panels[panel_id]["vis_var"].get()))

    def show_all_panels(self) -> None:
        for pid in self._panels:
            if not self._panels[pid]["visible"]:
                self.set_visible(pid, True)

    def hide_all_panels(self) -> None:
        for pid in self._panels:
            self.set_visible(pid, False)

    def popup_panel_menu(self, event=None, *, anchor: tk.Widget | None = None) -> None:
        if event is not None:
            x_root, y_root = event.x_root, event.y_root
        elif anchor is not None:
            anchor.update_idletasks()
            x_root = anchor.winfo_rootx()
            y_root = anchor.winfo_rooty() + anchor.winfo_height()
        else:
            self.update_idletasks()
            x_root = self.winfo_rootx() + max(0, self.winfo_width() // 2)
            y_root = self.winfo_rooty() + max(0, self.winfo_height() // 2)
        menu = tk.Menu(self, tearoff=0)
        for pid, p in self._panels.items():
            menu.add_checkbutton(
                label=p["title"],
                variable=p["vis_var"],
                command=lambda i=pid: self.on_panel_vis_toggle(i),
            )
        menu.add_separator()
        menu.add_command(label="Show all", command=self.show_all_panels)
        menu.add_command(label="Hide all", command=self.hide_all_panels)
        try:
            menu.tk_popup(x_root, y_root)
        finally:
            menu.grab_release()

    # ------------------------------------------------------------------ geometry

    def _grid_wh(self) -> tuple[int, int]:
        return max(1, self._grid.winfo_width()), max(1, self._grid.winfo_height())

    def _col_edges(self) -> list[int]:
        w, _ = self._grid_wh()
        edges = [round(c * w / self._cols) for c in range(self._cols)]
        edges.append(w)
        return edges

    def _row_edges(self) -> list[int]:
        _, h = self._grid_wh()
        edges = [round(r * h / self._rows) for r in range(self._rows)]
        edges.append(h)
        return edges

    def _clamp(self, row: int, col: int, rowspan: int, colspan: int) -> tuple[int, int, int, int]:
        rs = min(max(int(rowspan), 1), self._rows)
        cs = min(max(int(colspan), 1), self._cols)
        r = min(max(int(row), 0), self._rows - rs)
        c = min(max(int(col), 0), self._cols - cs)
        return r, c, rs, cs

    def _rect_px(self, pl: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        r, c, rs, cs = pl
        ce = self._col_edges()
        re = self._row_edges()
        x0, y0 = ce[c], re[r]
        x1, y1 = ce[c + cs], re[r + rs]
        return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)

    def _cell_at(self, x_root: int, y_root: int) -> tuple[int, int]:
        x = int(x_root - self._grid.winfo_rootx())
        y = int(y_root - self._grid.winfo_rooty())
        ce = self._col_edges()
        re = self._row_edges()
        c = self._cols - 1
        for ci in range(self._cols):
            if x < ce[ci + 1]:
                c = ci
                break
        r = self._rows - 1
        for ri in range(self._rows):
            if y < re[ri + 1]:
                r = ri
                break
        return r, c

    # ------------------------------------------------------------------ drawing

    def _on_grid_configure(self, event) -> None:
        if event.width < 2 or event.height < 2:
            return
        size = (event.width, event.height)
        if self._last_grid_size == size:
            return
        self._last_grid_size = size
        self._reposition_all_panels()
        self._draw_grid_lines()
        if self._op_id is None:
            self._notify_all_layouts()

    def _draw_grid_lines(self) -> None:
        self._grid_canvas.delete("grid")
        w, h = self._grid_wh()
        for x in self._col_edges()[1:-1]:
            self._grid_canvas.create_line(x, 0, x, h, fill="#3a3a3a", tags="grid")
        for y in self._row_edges()[1:-1]:
            self._grid_canvas.create_line(0, y, w, y, fill="#3a3a3a", tags="grid")
        if self._preview_placement is not None:
            self._draw_preview()
        self._sync_stack()
        self._grid_canvas.tag_lower("grid")

    def _draw_preview(self) -> None:
        self._grid_canvas.delete("preview")
        if self._preview_placement is None:
            return
        x0, y0, x1, y1 = self._rect_px(self._preview_placement)
        self._grid_canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#aac8ff",
            width=2,
            fill="#3366cc",
            stipple="gray25",
            tags="preview",
        )

    def _set_preview(self, pl: tuple[int, int, int, int]) -> None:
        self._preview_placement = self._clamp(*pl)
        self._draw_preview()

    def _clear_preview(self) -> None:
        self._preview_placement = None
        self._grid_canvas.delete("preview")

    # ------------------------------------------------------------------ panel placement

    def _place_panel(self, panel_id: str, *, raise_it: bool = False) -> None:
        p = self._panels[panel_id]
        shell = p["shell"]
        x0, y0, x1, y1 = self._rect_px(p["placement"])
        place_key = (x0, y0, x1 - x0, y1 - y0)
        if p.get("_last_place") != place_key or not shell.winfo_ismapped():
            p["_last_place"] = place_key
            shell.place(x=x0, y=y0, width=place_key[2], height=place_key[3])
        if raise_it:
            self.focus_panel(panel_id)

    def _reposition_all_panels(self) -> None:
        for pid, p in self._panels.items():
            if p["visible"] and pid != self._op_id:
                p.pop("_last_place", None)
                self._place_panel(pid)
        self._sync_stack()

    def _force_commit(self, panel_id: str, pl: tuple[int, int, int, int]) -> None:
        p = self._panels.get(panel_id)
        if p is None or not p["visible"]:
            return
        p["placement"] = self._clamp(*pl)
        p.pop("_last_place", None)
        self._place_panel(panel_id, raise_it=True)
        self.focus_panel(panel_id)
        self._grid.update_idletasks()
        self.after_idle(lambda pid=panel_id: self._notify_layout(pid))

    def _notify_layout(self, panel_id: str) -> None:
        if self._on_layout_changed is not None:
            self._on_layout_changed(panel_id)

    def _notify_all_layouts(self) -> None:
        if self._on_layout_changed is None:
            return
        if self._window_layout_after is not None:
            try:
                self.after_cancel(self._window_layout_after)
            except Exception:
                pass

        def _run() -> None:
            self._window_layout_after = None
            if self._on_layout_changed is None:
                return
            for pid, p in self._panels.items():
                if p["visible"]:
                    self._on_layout_changed(pid)

        self._window_layout_after = self.after_idle(_run)

    # ------------------------------------------------------------------ title drag

    def _on_title_press(self, event, panel_id: str) -> None:
        self._title_press = (event.x_root, event.y_root, panel_id)

    def _on_title_motion(self, event, panel_id: str) -> None:
        if self._op_id is not None:
            return
        if self._title_press is None or self._title_press[2] != panel_id:
            return
        x0, y0, _pid = self._title_press
        if abs(event.x_root - x0) + abs(event.y_root - y0) < self._DRAG_THRESHOLD_PX:
            return
        self._title_press = None
        self._start_move(event, panel_id)

    def _on_title_release(self, event, panel_id: str) -> None:
        self._title_press = None

    # ------------------------------------------------------------------ drag / resize

    def _start_op(self, panel_id: str, kind: str) -> None:
        p = self._panels.get(panel_id)
        if p is None or not p["visible"]:
            return
        self._op_id = panel_id
        self._op_kind = kind
        p["shell"].place_forget()
        self._set_preview(p["placement"])

    def _end_op(self) -> None:
        self._op_id = None
        self._op_kind = None
        self._resize_mode = None
        self._resize_anchor = None

    def _start_move(self, _event, panel_id: str) -> None:
        self.focus_panel(panel_id)
        self._start_op(panel_id, "move")

    def _start_resize(self, _event, panel_id: str, mode: str) -> None:
        self.focus_panel(panel_id)
        r, c, rs, cs = self._panels[panel_id]["placement"]
        self._resize_mode = mode
        self._resize_anchor = (r, c, r + rs - 1, c + cs - 1)
        self._start_op(panel_id, "resize")

    def _motion(self, event) -> None:
        if self._op_id is None:
            return
        if self._op_kind == "move":
            tr, tc = self._cell_at(event.x_root, event.y_root)
            _, _, rs, cs = self._panels[self._op_id]["placement"]
            self._set_preview((tr, tc, rs, cs))
        elif self._op_kind == "resize":
            self._set_preview(self._resize_to(event.x_root, event.y_root))

    def _release(self, _event) -> None:
        if self._op_id is None:
            return
        pid = self._op_id
        pl = self._preview_placement or self._panels[pid]["placement"]
        self._end_op()
        self._clear_preview()
        self._force_commit(pid, pl)
        self._draw_grid_lines()

    def _resize_to(self, x_root: int, y_root: int) -> tuple[int, int, int, int]:
        if self._resize_anchor is None or self._resize_mode is None or self._op_id is None:
            if self._op_id is not None:
                return self._panels[self._op_id]["placement"]
            return (0, 0, 1, 1)
        top, left, bottom, right = self._resize_anchor
        er, ec = self._cell_at(x_root, y_root)
        nr, nc, br, bc = top, left, bottom, right
        mode = self._resize_mode
        if "n" in mode:
            nr = min(max(er, 0), bottom)
        if "s" in mode:
            br = max(top, min(er, self._rows - 1))
        if "w" in mode:
            nc = min(max(ec, 0), right)
        if "e" in mode:
            bc = max(left, min(ec, self._cols - 1))
        return self._clamp(nr, nc, br - nr + 1, bc - nc + 1)

    def _add_grips(self, shell: tk.Frame, panel_id: str) -> None:
        for mode, cursor, rx, ry, anchor in self._RESIZE_EDGES:
            if mode in ("n", "s"):
                g = tk.Frame(shell, height=self._EDGE_PX, cursor=cursor, bg="#5a5a5a")
                g.place(relx=0.0, rely=ry, relwidth=1.0, anchor=anchor, height=self._EDGE_PX)
            else:
                g = tk.Frame(shell, width=self._EDGE_PX, cursor=cursor, bg="#5a5a5a")
                g.place(relx=rx, rely=0.0, relheight=1.0, anchor=anchor, width=self._EDGE_PX)
            g.bind("<ButtonPress-1>", lambda e, pid=panel_id, m=mode: self._start_resize(e, pid, m))
            g.lift()
        for mode, cursor, rx, ry, anchor in self._RESIZE_CORNERS:
            g = tk.Frame(shell, width=self._CORNER_PX, height=self._CORNER_PX, cursor=cursor, bg="#777777")
            g.place(relx=rx, rely=ry, anchor=anchor)
            g.bind("<ButtonPress-1>", lambda e, pid=panel_id, m=mode: self._start_resize(e, pid, m))
            g.lift()

    def _show_panel_menu(self, event) -> None:
        self.popup_panel_menu(event)

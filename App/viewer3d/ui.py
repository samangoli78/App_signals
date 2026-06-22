from __future__ import annotations

import copy
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser, filedialog

import numpy as np

from . import colormap as cmaplib
from .viewer import CartoMeshViewer
from ..ui.ribbon_widgets import (
    build_ribbon_shell,
    collapsible_section,
    ribbon_button,
    ribbon_checkbox,
    ribbon_label,
)
from ..ui.resize_pause import attach_viewer_resize_pause

class ColorbarSettingsDialog(tk.Toplevel):
    def __init__(self, viewer: CartoMeshViewer) -> None:
        super().__init__(viewer.winfo_toplevel())
        self.viewer = viewer
        self.title("Colorbar settings")
        self.transient(viewer.winfo_toplevel())
        self.resizable(True, True)
        self.attributes("-topmost", True)
        self._snapshot = viewer.snapshot_color_settings()

        self._knots = list(viewer.piece_knots)
        self._drag_knot_i: int | None = None
        self._custom_rows: list[dict] = []
        self._cust_drag_edge: int | None = None

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self._nb = nb

        self._tab_std = ttk.Frame(nb, padding=6)
        self._tab_custom = ttk.Frame(nb, padding=6)
        nb.add(self._tab_std, text="Standard colormap")
        nb.add(self._tab_custom, text=f"Custom (<={cmaplib.MAX_CUSTOM_BINS} bins)")

        self._build_standard_tab(self._tab_std)
        self._build_custom_tab(self._tab_custom)

        btns = ttk.Frame(self, padding=(8, 0, 8, 8))
        btns.pack(fill="x")
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side="right", padx=4)
        ttk.Button(btns, text="OK", command=self._on_ok).pack(side="right", padx=4)
        ttk.Button(btns, text="Apply", command=self._on_apply).pack(side="right", padx=4)

        self.bind("<Return>", lambda e: self._on_ok())
        self.bind("<Escape>", lambda e: self._on_cancel())
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.after(10, self._place_near_viewer)
        self._sync_custom_from_viewer()

    def _place_near_viewer(self) -> None:
        try:
            vx = self.viewer.winfo_rootx()
            vy = self.viewer.winfo_rooty()
            self.geometry(f"+{vx + 24}+{vy + 24}")
        except Exception:
            pass

    def _build_standard_tab(self, body: ttk.Frame) -> None:
        st = self.viewer.get_color_state()
        row = 0
        ttk.Label(body, text="Colormap").grid(row=row, column=0, sticky="w")
        self._cmap_var = tk.StringVar(value=st["cmap_name"])
        ttk.Combobox(body, textvariable=self._cmap_var, values=list(cmaplib.COLORMAPS), width=18, state="readonly").grid(
            row=row, column=1, sticky="w", padx=4
        )
        self._reverse_var = tk.BooleanVar(value=st["reverse"])
        ttk.Checkbutton(body, text="Reverse", variable=self._reverse_var).grid(row=row, column=2, sticky="w", padx=6)
        row += 1

        self._orient_var = tk.StringVar(value=self.viewer.cb_orientation)
        ttk.Label(body, text="Bar orientation").grid(row=row, column=0, sticky="w")
        of = ttk.Frame(body)
        of.grid(row=row, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(of, text="Vertical", value="vertical", variable=self._orient_var).pack(side="left")
        ttk.Radiobutton(of, text="Horizontal", value="horizontal", variable=self._orient_var).pack(side="left", padx=8)
        row += 1

        self._show_cb_var = tk.BooleanVar(value=self.viewer.show_colorbar)
        ttk.Checkbutton(body, text="Show on-screen colorbar", variable=self._show_cb_var).grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1

        ttk.Label(body, text="Number of bins").grid(row=row, column=0, sticky="w")
        self._bins_var = tk.IntVar(value=int(st["n_bins"]))
        ttk.Spinbox(body, from_=2, to=256, textvariable=self._bins_var, width=8, command=self._std_preview).grid(
            row=row, column=1, sticky="w", padx=4
        )
        ttk.Label(body, text="(2-256; 256 ~ smooth)").grid(row=row, column=2, sticky="w")
        row += 1

        self._auto_var = tk.BooleanVar(value=bool(st["auto_range"]))
        ttk.Checkbutton(body, text="Auto range (2-98 percentile)", variable=self._auto_var, command=self._on_auto).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        ttk.Label(body, text="Min").grid(row=row, column=0, sticky="w")
        self._vmin_var = tk.StringVar(value=f"{st['vmin']:.6g}")
        self._vmin_e = ttk.Entry(body, textvariable=self._vmin_var, width=14)
        self._vmin_e.grid(row=row, column=1, sticky="w", padx=4)
        ttk.Button(body, text="Snap min", command=self._snap_min).grid(row=row, column=2, sticky="w")
        row += 1
        ttk.Label(body, text="Max").grid(row=row, column=0, sticky="w")
        self._vmax_var = tk.StringVar(value=f"{st['vmax']:.6g}")
        self._vmax_e = ttk.Entry(body, textvariable=self._vmax_var, width=14)
        self._vmax_e.grid(row=row, column=1, sticky="w", padx=4)
        ttk.Button(body, text="Snap max", command=self._snap_max).grid(row=row, column=2, sticky="w")
        row += 1

        dmin, dmax = self.viewer.get_data_range()
        ttk.Label(body, text=f"Raw data: [{dmin:.4g}, {dmax:.4g}]   field: {st['scalar_field']}", foreground="#555").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 4)
        )
        row += 1

        ttk.Label(
            body,
            text="Knot lines (drag): remap how normalized values map through the colormap. "
            "Double-click empty space adds a knot; right-click a line removes it (max "
            f"{cmaplib.MAX_INTERNAL_KNOTS}).",
            wraplength=520,
        ).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        self._kcanvas = tk.Canvas(body, width=420, height=100, bg="#121218", highlightthickness=1, highlightbackground="#666")
        self._kcanvas.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        self._kcanvas.bind("<ButtonPress-1>", self._k_down)
        self._kcanvas.bind("<B1-Motion>", self._k_move)
        self._kcanvas.bind("<ButtonRelease-1>", self._k_up)
        self._kcanvas.bind("<Double-Button-1>", self._k_dbl)
        self._kcanvas.bind("<ButtonPress-3>", self._k_right)

        self._cmap_var.trace_add("write", lambda *a: self._std_preview())
        self._reverse_var.trace_add("write", lambda *a: self._std_preview())
        self._bins_var.trace_add("write", lambda *a: self._std_preview())
        self._vmin_var.trace_add("write", lambda *_: self._on_custom_range_var())
        self._vmax_var.trace_add("write", lambda *_: self._on_custom_range_var())
        self._on_auto()

    def _draw_custom_preview_safe(self) -> None:
        if hasattr(self, "_cust_prev"):
            self._draw_custom_preview()

    def _on_custom_range_var(self) -> None:
        if hasattr(self, "_custom_rows") and self._custom_rows:
            rng = self._cust_range()
            if rng is not None:
                lo, hi = rng
                self._custom_rows[0]["lo"].set(f"{lo:.6g}")
                self._custom_rows[-1]["hi"].set(f"{hi:.6g}")
        self._draw_custom_preview_safe()
        if hasattr(self, "_kcanvas"):
            self._std_preview()

    def _on_auto(self) -> None:
        st = "disabled" if self._auto_var.get() else "normal"
        self._vmin_e.configure(state=st)
        self._vmax_e.configure(state=st)
        if self._auto_var.get():
            lo, hi = self.viewer._resolved_range()
            self._vmin_var.set(f"{lo:.6g}")
            self._vmax_var.set(f"{hi:.6g}")
        self._std_preview()

    def _snap_min(self) -> None:
        d, _ = self.viewer.get_data_range()
        self._vmin_var.set(f"{d:.6g}")
        self._auto_var.set(False)
        self._on_auto()

    def _snap_max(self) -> None:
        _, d = self.viewer.get_data_range()
        self._vmax_var.set(f"{d:.6g}")
        self._auto_var.set(False)
        self._on_auto()

    def _k_x_from_t(self, t: float, pad: int, cw: int) -> float:
        return pad + t * (cw - 2 * pad)

    def _k_t_from_x(self, x: float, pad: int, cw: int) -> float:
        return float(np.clip((x - pad) / max(cw - 2 * pad, 1), 0.0, 1.0))

    def _std_preview(self, *_a) -> None:
        c = self._kcanvas
        c.delete("all")
        pad = 14
        cw = int(c["width"])
        ch = int(c["height"])
        mid = ch // 2
        h2 = 28
        y0, y1 = mid - h2 // 2, mid + h2 // 2
        n = 200
        ts = np.linspace(0, 1, n)
        name = self._cmap_var.get()
        if self._reverse_var.get() and not name.endswith("_r"):
            name = name + "_r"
        try:
            cm = cmaplib.get_cmap(name)
        except Exception:
            cm = cmaplib.get_cmap("jet")
        try:
            nb = max(2, min(256, int(self._bins_var.get())))
        except (tk.TclError, ValueError):
            nb = 256
        u = cmaplib.t_to_cmap_u(ts, self._knots)
        if nb < 256:
            lv = np.floor(u * nb).clip(0, nb - 1)
            u = (lv + 0.5) / nb
        cols = cm(u)
        for i in range(n - 1):
            xa = self._k_x_from_t(ts[i], pad, cw)
            xb = self._k_x_from_t(ts[i + 1], pad, cw)
            hx = cmaplib.rgb_to_hex(cols[i, :3])
            c.create_rectangle(xa, y0, xb, y1, fill=hx, outline=hx)
        c.create_rectangle(pad, y0, cw - pad, y1, outline="#999")
        for kt in sorted(self._knots):
            x = self._k_x_from_t(kt, pad, cw)
            c.create_line(x, y0 - 2, x, y1 + 2, fill="#ffcc00", width=2)

    def _k_pick(self, x: float) -> int | None:
        pad = 14
        cw = int(self._kcanvas["width"])
        knots = sorted(self._knots)
        best_i, best_d = None, 1e9
        for i, kt in enumerate(knots):
            xi = self._k_x_from_t(kt, pad, cw)
            d = abs(x - xi)
            if d < best_d and d < 8:
                best_d, best_i = d, i
        return best_i

    def _k_down(self, ev) -> None:
        self._knots = sorted(self._knots)
        self._drag_knot_i = self._k_pick(ev.x)

    def _k_move(self, ev) -> None:
        if self._drag_knot_i is None:
            return
        pad = 14
        cw = int(self._kcanvas["width"])
        knots = sorted(self._knots)
        i = self._drag_knot_i
        tnew = self._k_t_from_x(ev.x, pad, cw)
        lo = 1e-4
        hi = 1.0 - 1e-4
        if i > 0:
            lo = max(lo, knots[i - 1] + 1e-3)
        if i + 1 < len(knots):
            hi = min(hi, knots[i + 1] - 1e-3)
        tnew = float(np.clip(tnew, lo, hi))
        knots[i] = tnew
        self._knots = cmaplib.merge_knots01(knots)
        self._std_preview()

    def _k_up(self, _e) -> None:
        self._drag_knot_i = None

    def _k_dbl(self, ev) -> None:
        if self._k_pick(ev.x) is not None:
            return
        pad = 14
        cw = int(self._kcanvas["width"])
        tnew = self._k_t_from_x(ev.x, pad, cw)
        if len(self._knots) >= cmaplib.MAX_INTERNAL_KNOTS:
            messagebox.showinfo("Knots", f"At most {cmaplib.MAX_INTERNAL_KNOTS} internal knots.")
            return
        self._knots = cmaplib.merge_knots01(self._knots + [tnew])
        self._std_preview()

    def _k_right(self, ev) -> None:
        self._knots = sorted(self._knots)
        i = self._k_pick(ev.x)
        if i is None:
            return
        self._knots.pop(i)
        self._std_preview()

    def _build_custom_tab(self, body: ttk.Frame) -> None:
        row = 0
        ttk.Label(body, text="Bins (max 5)").grid(row=row, column=0, sticky="w")
        self._cust_n = tk.IntVar(value=3)
        ttk.Spinbox(
            body,
            from_=1,
            to=cmaplib.MAX_CUSTOM_BINS,
            textvariable=self._cust_n,
            width=4,
            command=self._rebuild_custom_rows,
        ).grid(row=row, column=1, sticky="w", padx=4)
        ttk.Button(body, text="Equal split on range", command=self._equal_split_custom).grid(
            row=row, column=2, sticky="w"
        )
        row += 1

        rng = ttk.Frame(body)
        rng.grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 4))
        ttk.Label(rng, text="Range min").grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(rng, textvariable=self._vmin_var, width=12).grid(row=0, column=1, padx=(0, 12))
        ttk.Label(rng, text="max").grid(row=0, column=2, padx=(0, 4))
        ttk.Entry(rng, textvariable=self._vmax_var, width=12).grid(row=0, column=3, padx=(0, 8))
        ttk.Label(
            rng,
            text="(defaults from current colorbar; drag bar edges below or edit here)",
            foreground="#666666",
        ).grid(row=0, column=4, sticky="w")
        row += 1

        self._cust_prev = tk.Canvas(
            body,
            width=420,
            height=56,
            bg="#121218",
            highlightthickness=1,
            highlightbackground="#666",
            cursor="hand2",
        )
        self._cust_prev.grid(row=row, column=0, columnspan=4, sticky="w", pady=6)
        self._cust_prev.bind("<ButtonPress-1>", self._cust_down)
        self._cust_prev.bind("<B1-Motion>", self._cust_move)
        self._cust_prev.bind("<ButtonRelease-1>", self._cust_up)
        self._cust_prev.bind("<ButtonPress-3>", self._cust_right)
        row += 1
        ttk.Label(
            body,
            text="Drag yellow borders to set bin edges. Right-click a bin to choose its color.",
            foreground="#666666",
            wraplength=520,
        ).grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1
        self._cust_grid = ttk.Frame(body)
        self._cust_grid.grid(row=row, column=0, columnspan=4, sticky="nw")
        self._rebuild_custom_rows()

    def _sync_custom_from_viewer(self) -> None:
        bins = self.viewer.custom_bins
        if bins:
            self._cust_n.set(min(cmaplib.MAX_CUSTOM_BINS, max(1, len(bins))))
        self._rebuild_custom_rows(from_viewer=True)

    def _rebuild_custom_rows(self, from_viewer: bool = False) -> None:
        for ch in self._cust_grid.winfo_children():
            ch.destroy()
        self._custom_rows.clear()
        n = int(self._cust_n.get())
        n = max(1, min(cmaplib.MAX_CUSTOM_BINS, n))

        existing: list[dict] | None = None
        if from_viewer and self.viewer.color_mode == "custom" and self.viewer.custom_bins:
            existing = sorted(copy.deepcopy(self.viewer.custom_bins), key=lambda b: float(b["lo"]))
            n = len(existing)
            self._cust_n.set(n)

        st = self.viewer.get_color_state()
        try:
            lo = float(self._vmin_var.get())
            hi = float(self._vmax_var.get())
        except (ValueError, tk.TclError):
            lo, hi = float(st["vmin"]), float(st["vmax"])
        if hi <= lo:
            hi = lo + 1.0
        palette = [(0.2, 0.45, 0.95), (0.15, 0.85, 0.35), (0.95, 0.75, 0.15), (0.95, 0.35, 0.2), (0.75, 0.35, 0.9)]
        for i in range(n):
            if existing and i < len(existing):
                b = existing[i]
                rlo, rhi, rgb = float(b["lo"]), float(b["hi"]), tuple(b["rgb"])
            else:
                edges = np.linspace(lo, hi, n + 1)
                rlo, rhi = float(edges[i]), float(edges[i + 1])
                rgb = palette[i % len(palette)]
            rowd = {"lo": tk.StringVar(value=f"{rlo:.6g}"), "hi": tk.StringVar(value=f"{rhi:.6g}"), "rgb": list(rgb)}
            self._custom_rows.append(rowd)
            fr = ttk.Frame(self._cust_grid)
            fr.grid(row=i, column=0, sticky="ew", pady=2)
            ttk.Label(fr, text=f"Bin {i+1}").grid(row=0, column=0, padx=2)
            ttk.Entry(fr, textvariable=rowd["lo"], width=10).grid(row=0, column=1, padx=2)
            ttk.Entry(fr, textvariable=rowd["hi"], width=10).grid(row=0, column=2, padx=2)
            ttk.Button(fr, text="Color…", command=lambda j=i: self._pick_color(j)).grid(row=0, column=3, padx=4)
            rowd["lo"].trace_add("write", lambda *_a: self._draw_custom_preview_safe())
            rowd["hi"].trace_add("write", lambda *_a: self._draw_custom_preview_safe())
        self._draw_custom_preview()

    def _cust_range(self) -> tuple[float, float] | None:
        try:
            lo = float(self._vmin_var.get())
            hi = float(self._vmax_var.get())
        except ValueError:
            return None
        if hi <= lo:
            return None
        return lo, hi

    def _cust_canvas_pad(self) -> int:
        return 10

    def _cust_val_to_x(self, val: float, lo: float, hi: float, w: int, pad: int) -> float:
        return pad + (val - lo) / (hi - lo) * (w - 2 * pad)

    def _cust_x_to_val(self, x: float, lo: float, hi: float, w: int, pad: int) -> float:
        t = (x - pad) / max(w - 2 * pad, 1)
        return float(lo + np.clip(t, 0.0, 1.0) * (hi - lo))

    def _cust_edge_values(self) -> list[float] | None:
        rng = self._cust_range()
        if rng is None or not self._custom_rows:
            return None
        lo, hi = rng
        edges = [lo]
        for row in self._custom_rows:
            try:
                edges.append(float(row["hi"].get()))
            except ValueError:
                return None
        edges[0] = lo
        edges[-1] = hi
        return edges

    def _cust_pick_edge(self, x: float) -> int | None:
        rng = self._cust_range()
        if rng is None:
            return None
        lo, hi = rng
        c = self._cust_prev
        w = int(c["width"])
        pad = self._cust_canvas_pad()
        edges = self._cust_edge_values()
        if edges is None:
            return None
        best_i, best_d = None, 1e9
        for i, ev in enumerate(edges):
            xi = self._cust_val_to_x(ev, lo, hi, w, pad)
            d = abs(x - xi)
            if d < best_d and d < 8:
                best_d, best_i = d, i
        return best_i

    def _cust_pick_bin(self, x: float) -> int | None:
        rng = self._cust_range()
        if rng is None or not self._custom_rows:
            return None
        lo, hi = rng
        c = self._cust_prev
        w = int(c["width"])
        pad = self._cust_canvas_pad()
        try:
            val = self._cust_x_to_val(x, lo, hi, w, pad)
        except Exception:
            return None
        for i, row in enumerate(self._custom_rows):
            try:
                a, b = float(row["lo"].get()), float(row["hi"].get())
            except ValueError:
                continue
            if a <= val <= b or (i == len(self._custom_rows) - 1 and abs(val - b) < 1e-9):
                return i
        return None

    def _cust_set_edge(self, edge_i: int, val: float) -> None:
        n = len(self._custom_rows)
        if n == 0:
            return
        rng = self._cust_range()
        if rng is None:
            return
        lo, hi = rng
        val = float(np.clip(val, lo, hi))
        eps = max((hi - lo) * 1e-4, 1e-9)

        if edge_i == 0:
            if n > 1:
                try:
                    next_lo = float(self._custom_rows[1]["lo"].get())
                except ValueError:
                    next_lo = hi
                val = min(val, next_lo - eps)
            val = max(val, lo)
            self._auto_var.set(False)
            self._vmin_var.set(f"{val:.6g}")
            self._custom_rows[0]["lo"].set(f"{val:.6g}")
            return

        if edge_i == n:
            if n > 1:
                try:
                    prev_hi = float(self._custom_rows[-2]["hi"].get())
                except ValueError:
                    prev_hi = lo
                val = max(val, prev_hi + eps)
            val = min(val, hi)
            self._auto_var.set(False)
            self._vmax_var.set(f"{val:.6g}")
            self._custom_rows[-1]["hi"].set(f"{val:.6g}")
            return

        if 1 <= edge_i < n:
            try:
                left_lo = float(self._custom_rows[edge_i - 1]["lo"].get())
                right_hi = float(self._custom_rows[edge_i]["hi"].get())
            except ValueError:
                return
            val = float(np.clip(val, left_lo + eps, right_hi - eps))
            self._custom_rows[edge_i - 1]["hi"].set(f"{val:.6g}")
            self._custom_rows[edge_i]["lo"].set(f"{val:.6g}")

    def _cust_down(self, ev) -> None:
        self._cust_drag_edge = self._cust_pick_edge(ev.x)

    def _cust_move(self, ev) -> None:
        if self._cust_drag_edge is None:
            return
        rng = self._cust_range()
        if rng is None:
            return
        lo, hi = rng
        w = int(self._cust_prev["width"])
        pad = self._cust_canvas_pad()
        val = self._cust_x_to_val(ev.x, lo, hi, w, pad)
        self._cust_set_edge(self._cust_drag_edge, val)
        self._draw_custom_preview()

    def _cust_up(self, _ev) -> None:
        self._cust_drag_edge = None

    def _cust_right(self, ev) -> None:
        bi = self._cust_pick_bin(ev.x)
        if bi is not None:
            self._pick_color(bi)

    def _pick_color(self, j: int) -> None:
        r, g, b = self._custom_rows[j]["rgb"]
        tup, hx = colorchooser.askcolor(color=cmaplib.rgb_to_hex((r, g, b)), parent=self)
        if tup:
            self._custom_rows[j]["rgb"] = [float(tup[0]) / 255.0, float(tup[1]) / 255.0, float(tup[2]) / 255.0]
            self._draw_custom_preview()

    def _equal_split_custom(self) -> None:
        try:
            lo = float(self._vmin_var.get())
            hi = float(self._vmax_var.get())
        except ValueError:
            messagebox.showerror("Custom", "Set valid min/max on the Standard tab first (or disable auto).")
            return
        if hi <= lo:
            messagebox.showerror("Custom", "Max must exceed min.")
            return
        n = int(self._cust_n.get())
        edges = np.linspace(lo, hi, n + 1)
        self._rebuild_custom_rows()
        for i in range(n):
            self._custom_rows[i]["lo"].set(f"{float(edges[i]):.6g}")
            self._custom_rows[i]["hi"].set(f"{float(edges[i+1]):.6g}")
        self._draw_custom_preview()

    def _draw_custom_preview(self) -> None:
        c = self._cust_prev
        c.delete("all")
        rng = self._cust_range()
        if rng is None:
            return
        lo, hi = rng
        w = int(c["width"])
        h = int(c["height"])
        pad = self._cust_canvas_pad()
        y0, y1 = 8, h - 14
        edges = self._cust_edge_values()
        if edges is None:
            return

        for i, row in enumerate(self._custom_rows):
            try:
                a, b = float(row["lo"].get()), float(row["hi"].get())
            except ValueError:
                continue
            xa = self._cust_val_to_x(a, lo, hi, w, pad)
            xb = self._cust_val_to_x(b, lo, hi, w, pad)
            if xb < xa:
                xa, xb = xb, xa
            r, g, b_ = row["rgb"]
            hx = cmaplib.rgb_to_hex((r, g, b_))
            c.create_rectangle(xa, y0, xb, y1, fill=hx, outline="#ccc", tags=("bin", f"bin{i}"))

        c.create_rectangle(pad, y0, w - pad, y1, outline="#888")
        for i, ev in enumerate(edges):
            x = self._cust_val_to_x(ev, lo, hi, w, pad)
            c.create_line(x, y0 - 2, x, y1 + 2, fill="#ffcc00", width=2, tags=("edge", f"edge{i}"))
            if 0 < i < len(edges) - 1:
                c.create_text(x, y0 - 4, text=f"{ev:.4g}", anchor="s", fill="#ccc", font=("Segoe UI", 8))

        c.create_text(pad, h - 2, text=f"min {lo:.4g}", anchor="sw", fill="#aaa", font=("Segoe UI", 8))
        c.create_text(w - pad, h - 2, text=f"max {hi:.4g}", anchor="se", fill="#aaa", font=("Segoe UI", 8))

    def _read_std(self) -> dict | None:
        try:
            n_bins = int(self._bins_var.get())
            vmin = float(self._vmin_var.get())
            vmax = float(self._vmax_var.get())
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Colorbar", str(exc))
            return None
        if n_bins < 2 or n_bins > 256:
            messagebox.showerror("Colorbar", "Bins must be 2–256.")
            return None
        if not self._auto_var.get() and vmax <= vmin:
            messagebox.showerror("Colorbar", "Max must be greater than min.")
            return None
        return {
            "cmap_name": self._cmap_var.get(),
            "reverse": bool(self._reverse_var.get()),
            "n_bins": n_bins,
            "auto_range": bool(self._auto_var.get()),
            "vmin": vmin,
            "vmax": vmax,
            "piece_knots": cmaplib.merge_knots01(list(self._knots)),
            "cb_orientation": self._orient_var.get(),
            "show_colorbar": bool(self._show_cb_var.get()),
        }

    def _read_custom_bins(self) -> list[dict] | None:
        bins: list[dict] = []
        for row in self._custom_rows:
            try:
                lo = float(row["lo"].get())
                hi = float(row["hi"].get())
            except ValueError:
                messagebox.showerror("Custom", "Each bin needs numeric low / high.")
                return None
            if hi <= lo:
                messagebox.showerror("Custom", "Each bin needs high > low.")
                return None
            r, g, b = row["rgb"]
            bins.append({"lo": lo, "hi": hi, "rgb": [float(r), float(g), float(b)]})
        bins.sort(key=lambda b: b["lo"])
        for i in range(len(bins) - 1):
            if bins[i + 1]["lo"] < bins[i]["hi"] - 1e-9:
                messagebox.showerror("Custom", "Bins must not overlap (sort by low edge).")
                return None
        return bins

    def _on_apply(self) -> bool:
        std = self._read_std()
        if std is None:
            return False
        try:
            sel = self._nb.index(self._nb.select())
        except tk.TclError:
            sel = 0
        if sel == 0:
            self.viewer.apply_color_settings(
                cmap_name=std["cmap_name"],
                reverse=std["reverse"],
                n_bins=std["n_bins"],
                auto_range=std["auto_range"],
                vmin=None if std["auto_range"] else std["vmin"],
                vmax=None if std["auto_range"] else std["vmax"],
                color_mode="standard",
                piece_knots=std["piece_knots"],
                custom_bins=[],
                cb_orientation=std["cb_orientation"],
                show_colorbar=std["show_colorbar"],
            )
        else:
            bins = self._read_custom_bins()
            if bins is None:
                return False
            self.viewer.apply_color_settings(
                cmap_name=std["cmap_name"],
                reverse=std["reverse"],
                n_bins=std["n_bins"],
                auto_range=std["auto_range"],
                vmin=None if std["auto_range"] else std["vmin"],
                vmax=None if std["auto_range"] else std["vmax"],
                color_mode="custom",
                piece_knots=[],
                custom_bins=bins,
                cb_orientation=std["cb_orientation"],
                show_colorbar=std["show_colorbar"],
            )
        return True

    def _on_ok(self) -> None:
        if self._on_apply():
            self.destroy()

    def _on_cancel(self) -> None:
        self.viewer.restore_color_settings(self._snapshot)
        self.destroy()


class SphereSettingsDialog(tk.Toplevel):
    """Small dialog to control electrode sphere size and projection mode."""

    def __init__(self, viewer: CartoMeshViewer) -> None:
        super().__init__(viewer.winfo_toplevel())
        self.viewer = viewer
        self.title("Electrode spheres")
        self.transient(viewer.winfo_toplevel())
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self._snapshot = {
            "factor": float(viewer.sphere_radius_factor),
            "use_projected": bool(viewer.use_projected),
        }

        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Sphere size").grid(row=0, column=0, sticky="w")
        self._size_var = tk.DoubleVar(value=float(viewer.sphere_radius_factor))
        self._size_scale = ttk.Scale(
            body,
            from_=0.2,
            to=4.0,
            orient="horizontal",
            variable=self._size_var,
            command=self._on_size_change,
            length=240,
        )
        self._size_scale.grid(row=0, column=1, sticky="ew", padx=6)
        self._size_lbl = ttk.Label(body, text=f"{float(self._size_var.get()):.2f}x")
        self._size_lbl.grid(row=0, column=2, padx=(6, 0))

        ttk.Label(body, text="Position").grid(row=1, column=0, sticky="w", pady=(8, 0))
        pf = ttk.Frame(body)
        pf.grid(row=1, column=1, columnspan=2, sticky="w", pady=(8, 0))
        self._pos_var = tk.StringVar(value="projected" if viewer.use_projected else "original")
        ttk.Radiobutton(
            pf,
            text="Projected on mesh (closest surface point)",
            value="projected",
            variable=self._pos_var,
            command=self._on_pos_change,
        ).pack(anchor="w")
        ttk.Radiobutton(
            pf,
            text="Original (raw electrode coordinates)",
            value="original",
            variable=self._pos_var,
            command=self._on_pos_change,
        ).pack(anchor="w")

        ttk.Button(body, text="Re-project to mesh", command=self._on_reproject).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(fill="x")
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side="right", padx=4)
        ttk.Button(btns, text="OK", command=self.destroy).pack(side="right", padx=4)

        self.bind("<Return>", lambda e: self.destroy())
        self.bind("<Escape>", lambda e: self._on_cancel())
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(10, self._place_near_viewer)

    def _place_near_viewer(self) -> None:
        try:
            vx = self.viewer.winfo_rootx()
            vy = self.viewer.winfo_rooty()
            self.geometry(f"+{vx + 24}+{vy + 24}")
        except Exception:
            pass

    def _on_size_change(self, _v: str | None = None) -> None:
        try:
            f = float(self._size_var.get())
        except (tk.TclError, ValueError):
            return
        self._size_lbl.config(text=f"{f:.2f}x")
        self.viewer.set_sphere_radius_factor(f)

    def _on_pos_change(self) -> None:
        self.viewer.set_use_projected(self._pos_var.get() == "projected")

    def _on_reproject(self) -> None:
        self.viewer.recompute_projection()

    def _on_cancel(self) -> None:
        self.viewer.set_sphere_radius_factor(self._snapshot["factor"])
        self.viewer.set_use_projected(self._snapshot["use_projected"])
        self.destroy()


class InterpolateSettingsDialog(tk.Toplevel):
    """Real-time geodesic-radius dialog for the interpolation cut size.

    Sliding ``r`` updates the viewer immediately so the user can preview the
    cut size as they drag. The Global toggle lives on the toolbar (outside
    this window) so the user can flip modes without re-opening the dialog.

    OK keeps the previewed radius; Cancel restores the snapshot taken at
    open-time so accidental experiments don't leave the viewer in a strange
    state.
    """

    def __init__(self, viewer: CartoMeshViewer) -> None:
        super().__init__(viewer.winfo_toplevel())
        self.viewer = viewer
        self.title("Interpolation settings")
        self.transient(viewer.winfo_toplevel())
        self.resizable(False, False)
        self.attributes("-topmost", True)
        # Snapshot for Cancel.
        self._snapshot = {
            "radius": (
                float(viewer.interpolation_radius)
                if viewer.interpolation_radius is not None
                else 0.0
            ),
        }

        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)

        # Suggest a sensible default range based on the mesh's mean edge.
        try:
            default_r = float(viewer.default_interpolation_radius() or 0.0)
        except Exception:
            default_r = 0.0
        if default_r <= 0:
            default_r = 1.0
        max_r = max(default_r * 20.0, 1.0)

        ttk.Label(body, text="Geodesic radius r:").grid(row=0, column=0, sticky="w")
        r_init = float(self._snapshot["radius"])
        if r_init <= 0:
            r_init = default_r
        self._r_var = tk.DoubleVar(value=r_init)
        self._r_scale = ttk.Scale(
            body,
            from_=0.0,
            to=max_r,
            orient="horizontal",
            variable=self._r_var,
            command=self._on_radius_drag,
            length=260,
        )
        self._r_scale.grid(row=0, column=1, sticky="ew", padx=6)

        self._r_entry_var = tk.StringVar(value=f"{r_init:.3f}")
        self._r_entry = ttk.Spinbox(
            body,
            from_=0.0,
            to=max_r * 4,  # let the user type beyond the slider tail
            increment=max(default_r * 0.1, 0.1),
            width=8,
            textvariable=self._r_entry_var,
            command=self._on_radius_entry,
        )
        self._r_entry.grid(row=0, column=2, sticky="w")
        self._r_entry.bind("<Return>", lambda _e: self._on_radius_entry())
        self._r_entry.bind("<FocusOut>", lambda _e: self._on_radius_entry())

        ttk.Label(
            body,
            text=f"(0 = use default ≈ 10× mean edge ≈ {default_r:.3f})",
            foreground="#666666",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 8))

        ttk.Label(
            body,
            text=(
                "Interpolation minimises Dirichlet energy on the geodesic patch "
                "inside this radius (known values only at electrodes)."
            ),
            foreground="#666666",
        ).grid(row=2, column=0, columnspan=3, sticky="w")

        # Push the initial radius to the viewer if it differed from snapshot.
        self._apply_radius(r_init)

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(fill="x")
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side="right", padx=4)
        ttk.Button(btns, text="OK", command=self._on_ok).pack(side="right", padx=4)

        self.bind("<Return>", lambda e: self._on_ok())
        self.bind("<Escape>", lambda e: self._on_cancel())
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.after(10, self._place_near_viewer)

    # ------------------------------------------------------------------ events
    def _place_near_viewer(self) -> None:
        try:
            vx = self.viewer.winfo_rootx()
            vy = self.viewer.winfo_rooty()
            self.geometry(f"+{vx + 24}+{vy + 24}")
        except Exception:
            pass

    def _on_radius_drag(self, _v: str | None = None) -> None:
        try:
            r = float(self._r_var.get())
        except (tk.TclError, ValueError):
            return
        self._r_entry_var.set(f"{r:.3f}")
        self._apply_radius(r)

    def _on_radius_entry(self) -> None:
        raw = self._r_entry_var.get()
        try:
            r = float(raw)
        except (TypeError, ValueError):
            return
        r = max(0.0, r)
        try:
            self._r_var.set(r)
        except tk.TclError:
            pass
        self._apply_radius(r)

    def _apply_radius(self, r: float) -> None:
        self.viewer.set_interpolation_radius(r if r > 0 else None)

    def _on_ok(self) -> None:
        # Current preview = the committed value. Just close.
        self.destroy()

    def _on_cancel(self) -> None:
        try:
            r = float(self._snapshot["radius"])
        except (TypeError, ValueError):
            r = 0.0
        self.viewer.set_interpolation_radius(r if r > 0 else None)
        self.destroy()


class CartoMeshPanel(tk.Frame):
    def __init__(self, master, carto, scalar_field: str = "bipolar", cmap_name: str = "jet", **kw) -> None:
        super().__init__(master, **kw)
        self.configure(background="black")

        shell = build_ribbon_shell(self, title="3D mesh", default_width=200)
        self.ribbon_pane = shell["outer_pane"]
        col = shell["ribbon_column"]

        display = collapsible_section(col, "Display")
        ribbon_label(display["body"], "Scalar field")
        self._field_var = tk.StringVar(value=scalar_field)
        self._field_combo = ttk.Combobox(
            display["body"],
            textvariable=self._field_var,
            values=list(cmaplib.SCALAR_FIELDS),
            state="readonly",
        )
        self._field_combo.pack(fill="x", padx=6, pady=2)
        self._field_combo.bind("<<ComboboxSelected>>", self._on_field_change)

        ribbon_label(display["body"], "Colormap")
        self._cmap_var = tk.StringVar(value=cmap_name)
        self._cmap_combo = ttk.Combobox(
            display["body"],
            textvariable=self._cmap_var,
            values=list(cmaplib.COLORMAPS),
            state="readonly",
        )
        self._cmap_combo.pack(fill="x", padx=6, pady=2)
        self._cmap_combo.bind("<<ComboboxSelected>>", self._on_cmap_change)

        view = collapsible_section(col, "View")
        ribbon_button(view["body"], "Reset view", self._on_reset)
        ribbon_button(view["body"], "Colorbar…", self._open_colorbar_settings)
        ribbon_button(view["body"], "Spheres…", self._open_sphere_settings)
        self._show_cb_var = tk.IntVar(value=1)
        ribbon_checkbox(view["body"], "Show colorbar", self._show_cb_var, self._toggle_colorbar)

        interp = collapsible_section(col, "Interpolation")
        self._interp_var = tk.IntVar(value=0)
        ribbon_checkbox(interp["body"], "Interpolate", self._interp_var, self._toggle_interpolate)
        ribbon_button(interp["body"], "Interpolate…", self._open_interp_settings)
        self._global_var = tk.IntVar(value=0)
        ribbon_checkbox(interp["body"], "Global harmonic", self._global_var, self._toggle_global)
        self._classic_mesh_var = tk.IntVar(value=1)
        ribbon_checkbox(
            interp["body"],
            "Classic mesh fill",
            self._classic_mesh_var,
            self._toggle_classic_mesh_fill,
        )

        export = collapsible_section(col, "Export", expanded=False)
        ribbon_button(export["body"], "Export VTK…", self._export_vtk_deltas)

        self.viewer = CartoMeshViewer(
            shell["content_host"], carto, scalar_field=scalar_field, cmap_name=cmap_name
        )
        self.viewer.pack(fill="both", expand=True)
        self.viewer.add_color_listener(self._on_viewer_colors)
        self.viewer.add_fields_listener(self._on_viewer_fields)
        self._interp_dialog: "InterpolateSettingsDialog | None" = None

        attach_viewer_resize_pause(
            self.viewer,
            panes=[shell["outer_pane"]],
            watch_widgets=[shell["ribbon_shell"], shell["content_host"], self.viewer],
        )

    def _on_viewer_colors(self, _viewer) -> None:
        self._show_cb_var.set(1 if self.viewer.show_colorbar else 0)

    def _on_field_change(self, _e=None) -> None:
        self.viewer.set_scalar_field(self._field_var.get())

    def _on_cmap_change(self, _e=None) -> None:
        self.viewer.apply_color_settings(
            cmap_name=self._cmap_var.get(),
            color_mode="standard",
            custom_bins=[],
        )

    def _on_reset(self) -> None:
        self.viewer.reset_view()

    def _open_colorbar_settings(self) -> None:
        self.viewer.open_colorbar_settings()

    def _open_sphere_settings(self) -> None:
        self.viewer.open_sphere_settings()

    def _toggle_colorbar(self) -> None:
        self.viewer.show_colorbar = bool(self._show_cb_var.get())
        self.viewer._request_redraw()

    def _toggle_interpolate(self) -> None:
        self.viewer.set_interpolation_enabled(bool(self._interp_var.get()))
        self._refresh_field_choices()

    def _open_interp_settings(self) -> None:
        # Re-focus an open dialog instead of stacking duplicates.
        if self._interp_dialog is not None and self._interp_dialog.winfo_exists():
            try:
                self._interp_dialog.lift()
                self._interp_dialog.focus_set()
                return
            except tk.TclError:
                self._interp_dialog = None
        self._interp_dialog = InterpolateSettingsDialog(self.viewer)

    def _toggle_global(self) -> None:
        self.viewer.use_global_patch_harmonic = bool(self._global_var.get())
        if (
            self.viewer.interpolation_enabled
            and self.viewer._mesh_loaded
            and str(self.viewer.scalar_field).startswith("delta:")
        ):
            self.viewer._compute_delta_interpolated()
        self.viewer._request_redraw()

    def _toggle_classic_mesh_fill(self) -> None:
        self.viewer.prefer_legacy_mesh_rendering = bool(self._classic_mesh_var.get())
        if not self.viewer.prefer_legacy_mesh_rendering:
            try:
                self.viewer._mesh_gl_build_or_refresh(static=True)
            except Exception:
                traceback.print_exc()
        self.viewer._request_redraw()

    def _export_vtk_deltas(self) -> None:
        dlg = tk.Toplevel(self.winfo_toplevel())
        dlg.title("Export VTK for Carto")
        dlg.transient(self.winfo_toplevel())
        dlg.resizable(False, False)
        dlg.grab_set()

        body = ttk.Frame(dlg, padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body,
            text="Legacy Carto .vtk (version 4.1):\n"
            "PatientData header, SCALARS scalars double [0–1],\n"
            "LOOKUP_TABLE lookup_table (1000 rows, 4 color bands), NORMALS Normals float.\n"
            "Triangle winding is swapped for Carto-compatible face orientation.",
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(body, text="Patient name:").pack(anchor="w")
        name_var = tk.StringVar()
        ttk.Entry(body, textvariable=name_var, width=48).pack(fill="x", pady=(4, 10))

        ttk.Label(
            body,
            text="Interpolation radius / Global harmonic (toolbar) affect scalar values only.",
            foreground="#444",
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        btn_row = ttk.Frame(body)
        btn_row.pack(fill="x")

        def _cancel() -> None:
            dlg.grab_release()
            dlg.destroy()

        def _go() -> None:
            patient_name = str(name_var.get()).strip()
            if not patient_name:
                messagebox.showwarning("VTK export", "Patient name is required.", parent=dlg)
                return
            dlg.grab_release()
            dlg.destroy()
            folder = filedialog.askdirectory(
                title="Export Carto .vtk files — choose folder",
                parent=self.winfo_toplevel(),
            )
            if not folder:
                return
            try:
                n = int(self.viewer.export_vtk_deltas(folder, patient_name=patient_name))
                if n <= 0:
                    messagebox.showwarning(
                        "VTK export",
                        "No files written. Load a mesh and compute delta metrics first.",
                        parent=self.winfo_toplevel(),
                    )
                    return
                messagebox.showinfo(
                    "VTK export",
                    f"Wrote {n} Carto .vtk file(s).\n\nPatient: {patient_name}\nFolder: {folder}",
                    parent=self.winfo_toplevel(),
                )
            except Exception as exc:
                traceback.print_exc()
                messagebox.showerror("VTK export", str(exc), parent=self.winfo_toplevel())

        ttk.Button(btn_row, text="Cancel", command=_cancel).pack(side="right")
        ttk.Button(btn_row, text="Choose folder && export…", command=_go).pack(side="right", padx=(0, 8))
        dlg.bind("<Escape>", lambda _e: _cancel())

    def _on_viewer_fields(self, _viewer) -> None:
        # Fired when the viewer learns about new delta metrics. Sync the combo.
        self._refresh_field_choices()

    def _refresh_field_choices(self) -> None:
        try:
            fields = list(self.viewer.available_fields())
        except Exception:
            fields = list(cmaplib.SCALAR_FIELDS)
        self._field_combo["values"] = fields
        cur = self._field_var.get()
        if cur not in fields:
            self._field_var.set(fields[0] if fields else "bipolar")



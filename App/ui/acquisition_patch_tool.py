"""Acquisition patch analysis — settings window + main Carto 3D mesh display.

Harmonic fields are shown on the main ``CartoMeshViewer``. Global colorbar
limits are computed from all time samples in the patch (background thread).
"""

from __future__ import annotations

import queue
import threading
import traceback
from typing import TYPE_CHECKING, Callable

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from ..triple_extra import Triple_Extra
from ..viewer3d import colormap as cm
from ..viewer3d import laplacian as lap

if TYPE_CHECKING:
    from ..main_app import App
    from ..viewer3d.viewer import CartoMeshViewer


def _resolve_uni_col(signals, name: str) -> str:
    from ..plotting.presenter import PlotPresenter

    return PlotPresenter._resolve_channel_name(signals, str(name))


def _resolve_channel(signals, name: str) -> str:
    from ..plotting.presenter import PlotPresenter

    return PlotPresenter._resolve_channel_name(signals, str(name))


def _resolve_ref_col(signals, ref_channel: str | None) -> str | None:
    from ..plotting.presenter import PlotPresenter

    try:
        return PlotPresenter._resolve_first_existing(
            signals, ["V5", ref_channel, "CS1", "M4", "M3"]
        )
    except Exception:
        return None


def _decimate_xy(x: np.ndarray, y: np.ndarray, max_pts: int = 12_000) -> tuple[np.ndarray, np.ndarray]:
    from ..plotting.presenter import _decimate_xy as _dec

    return _dec(x, y, max_pts=max_pts)


def _compute_dvdt_mv_per_ms(S: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Time derivative in mV/ms (Carto index is seconds; typical fs ≈ 1 kHz)."""
    S = np.asarray(S, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64).ravel()
    if S.size == 0:
        return np.zeros_like(S)
    t_ms = t_s * 1000.0
    out = np.zeros_like(S)
    if t_ms.size < 2:
        return out
    for j in range(S.shape[1]):
        out[:, j] = np.gradient(S[:, j], t_ms)
    return out


def _mesh_viewer(app: "App") -> "CartoMeshViewer | None":
    panel = getattr(app, "mesh_panel", None)
    if panel is None:
        return None
    return getattr(panel, "viewer", None)


def _acquisition_stack(
    app: "App", sec_idx: int
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    section = app.carto.cont[int(sec_idx)]
    df_pt = section[0]
    signals = section[2]
    cols: list[str] = []
    series: list[np.ndarray] = []
    pos: list[tuple[float, float, float]] = []
    for j in range(len(df_pt)):
        row = df_pt.iloc[j]
        uni = row["unipolar"]
        try:
            cname = _resolve_uni_col(signals, str(uni))
        except Exception:
            continue
        series.append(np.asarray(signals[cname].values, dtype=np.float64))
        cols.append(str(uni))
        pos.append((float(row["x"]), float(row["y"]), float(row["z"])))
    if not series:
        return np.array([], dtype=np.float64), np.zeros((0, 0), dtype=np.float64), [], np.zeros((0, 3))
    t = np.asarray(signals.index, dtype=np.float64)
    n = min(s.shape[0] for s in series)
    S = np.column_stack([s[:n] for s in series])
    t = t[:n]
    P = np.asarray(pos, dtype=np.float64)
    return t, S, cols, P


def _reference_traces(
    app: "App", sec_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    """M1/M2 intracardiac reference (same M1−M2 diff as main plot), time in ms."""
    section = app.carto.cont[int(sec_idx)]
    signals = section[2]
    df_pt = section[0]
    t_s = np.asarray(signals.index, dtype=np.float64)
    t_ms = t_s * 1000.0

    for hi_name, lo_name in (("M1", "M2"), ("M", "M2")):
        try:
            hi_col = _resolve_channel(signals, hi_name)
            lo_col = _resolve_channel(signals, lo_name)
            y_hi = np.asarray(signals[hi_col].values, dtype=np.float64)
            y_lo = np.asarray(signals[lo_col].values, dtype=np.float64)
            n = min(t_ms.size, y_hi.size, y_lo.size)
            if n <= 0:
                continue
            return (
                t_ms[:n],
                (y_hi[:n] - y_lo[:n]),
                y_hi[:n],
                y_lo[:n],
                f"{hi_name} − {lo_name} (mV)",
            )
        except Exception:
            continue

    ref_meta = None
    try:
        ref_meta = str(df_pt.iloc[0]["refference_chanel"])
    except Exception:
        pass
    col = _resolve_ref_col(signals, ref_meta)
    if col is None:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            None,
            None,
            "reference",
        )
    y = np.asarray(signals[col].values, dtype=np.float64)
    n = min(t_ms.size, y.size)
    return t_ms[:n], y[:n], None, None, f"{col} (mV)"


def _pad_range(vmin: float, vmax: float) -> tuple[float, float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return 0.0, 1.0
    pad = 0.02 * (vmax - vmin)
    return vmin - pad, vmax + pad


def _merge_bool_mask_runs_ms(t_ms: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous True runs in *mask* as (t_start_ms, t_end_ms) intervals."""
    t_ms = np.asarray(t_ms, dtype=np.float64).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()
    n = min(t_ms.size, mask.size)
    if n <= 0:
        return []
    out: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        out.append((float(t_ms[i]), float(t_ms[j - 1])))
        i = j
    return out


def _interest_sample_mask(
    t_s: np.ndarray,
    ref_y: np.ndarray,
    *,
    stim_pad_ms: float = 100.0,
    sr_pad_ms: float = 50.0,
    fs: float = 1000.0,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Union of Triple-Extra stim windows (±stim_pad) and SR windows (±sr_pad only)."""
    t_s = np.asarray(t_s, dtype=np.float64).ravel()
    ref_y = np.asarray(ref_y, dtype=np.float64).ravel()
    n = min(t_s.size, ref_y.size)
    if n <= 0:
        return np.zeros(0, dtype=bool), []
    t_s = t_s[:n]
    ref_y = ref_y[:n]
    t_ms = t_s * 1000.0
    stim_pad_n = max(1, int(round(float(stim_pad_ms) * fs / 1000.0)))
    sr_pad_n = max(1, int(round(float(sr_pad_ms) * fs / 1000.0)))

    stim = ref_y.copy()
    try:
        egm = Triple_Extra(t_s, None, T=float(t_s[-1]) if t_s.size else 2.5, fs=fs)
        info = egm.compute_windows(stimulation=stim, reference=ref_y, margin=0)
    except Exception:
        traceback.print_exc()
        mask = np.ones(n, dtype=bool)
        return mask, [(float(t_ms[0]), float(t_ms[-1]))]

    mask = np.zeros(n, dtype=bool)
    for s, dur in zip(info.get("stim_start") or [], info.get("stim_duration") or []):
        lo = max(0, int(s) - stim_pad_n)
        hi = min(n, int(s) + int(dur) + stim_pad_n)
        if hi > lo:
            mask[lo:hi] = True
    for s, dur in zip(info.get("sinus_start") or [], info.get("sinus_duration") or []):
        lo = max(0, int(s) - sr_pad_n)
        hi = min(n, int(s) + int(dur) + sr_pad_n)
        if hi > lo:
            mask[lo:hi] = True

    if not mask.any():
        mask[:] = True
    return mask, _merge_bool_mask_runs_ms(t_ms, mask)


def _global_ranges_worker(
    L,
    graph,
    V: np.ndarray,
    anchor_idx: np.ndarray,
    mask: np.ndarray,
    S: np.ndarray,
    dSdt: np.ndarray,
    t_s: np.ndarray,
    local_radius: float,
    sample_mask: np.ndarray | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Global mesh color limits + dV/ds series (only *sample_mask* time indices)."""
    S = np.asarray(S, dtype=np.float64)
    dSdt = np.asarray(dSdt, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    anchor_idx = np.asarray(anchor_idx, dtype=np.int64).ravel()
    n = int(S.shape[0])
    n_anc = int(anchor_idx.size)
    patch = np.asarray(mask, dtype=bool)
    t_ms = np.asarray(t_s, dtype=np.float64).ravel() * 1000.0
    sm = np.ones(n, dtype=bool) if sample_mask is None else np.asarray(sample_mask, dtype=bool).ravel()[:n]
    if sm.size < n:
        sm = np.pad(sm, (0, n - sm.size), constant_values=False)
    ks = [k for k in range(n) if sm[k]]
    if not ks:
        ks = list(range(n))

    vmin_a, vmax_a = np.inf, -np.inf
    vmin_d, vmax_d = np.inf, -np.inf
    vmin_s, vmax_s = np.inf, -np.inf
    vmin_dd, vmax_dd = np.inf, -np.inf

    dvds_mag_series = np.full((n, n_anc), np.nan, dtype=np.float64)
    vec_dirs_series = np.full((n, n_anc, 3), np.nan, dtype=np.float64)
    n_pass = max(1, len(ks))

    for ii, k in enumerate(ks):
        if progress is not None:
            progress(ii, n_pass, "Global limits + dV/ds (windows of interest)")
        vals = np.asarray(S[k, :], dtype=np.float64).ravel()
        phi = lap.harmonic_interpolate(L, anchor_idx, vals, free_mask=mask)
        fin = phi[patch & np.isfinite(phi)]
        if fin.size:
            vmin_a = min(vmin_a, float(np.min(fin)))
            vmax_a = max(vmax_a, float(np.max(fin)))

        vals_d = np.asarray(dSdt[k, :], dtype=np.float64).ravel()
        phi_d = lap.harmonic_interpolate(L, anchor_idx, vals_d, free_mask=mask)
        fin_d = phi_d[patch & np.isfinite(phi_d)]
        if fin_d.size:
            vmin_d = min(vmin_d, float(np.min(fin_d)))
            vmax_d = max(vmax_d, float(np.max(fin_d)))

        _o, _d, mags = lap.max_path_dvds_per_anchor(
            V,
            graph,
            anchor_idx,
            phi,
            local_radius=local_radius,
            patch_mask=mask,
        )
        dvds_mag_series[k, :] = mags
        vec_dirs_series[k, :, :] = _d
        mags_fill = np.where(np.isfinite(mags), mags, 0.0)
        phi_s = lap.harmonic_interpolate(L, anchor_idx, mags_fill, free_mask=mask)
        fin_s = phi_s[patch & np.isfinite(phi_s)]
        if fin_s.size:
            vmin_s = min(vmin_s, float(np.min(fin_s)))
            vmax_s = max(vmax_s, float(np.max(fin_s)))

    d_mag_dt_series = np.full((n, n_anc), np.nan, dtype=np.float64)
    if n_anc > 0 and n > 1 and t_ms.size >= n:
        for e in range(n_anc):
            col = dvds_mag_series[:, e].copy()
            if np.any(np.isfinite(col)):
                col[~sm] = np.nan
                d_mag_dt_series[:, e] = np.gradient(np.nan_to_num(col, nan=0.0), t_ms[:n])

    for ii, k in enumerate(ks):
        if progress is not None:
            progress(ii, n_pass, "Global limits d(dV/ds)/dt")
        row = np.where(np.isfinite(d_mag_dt_series[k, :]), d_mag_dt_series[k, :], 0.0)
        phi_dd = lap.harmonic_interpolate(L, anchor_idx, row, free_mask=mask)
        fin_dd = phi_dd[patch & np.isfinite(phi_dd)]
        if fin_dd.size:
            vmin_dd = min(vmin_dd, float(np.min(fin_dd)))
            vmax_dd = max(vmax_dd, float(np.max(fin_dd)))

    return {
        "cb_amp": _pad_range(vmin_a, vmax_a),
        "cb_dv": _pad_range(vmin_d, vmax_d),
        "cb_dvds": _pad_range(vmin_s, vmax_s),
        "cb_ddvds_dt": _pad_range(vmin_dd, vmax_dd),
        "dvds_mag_series": dvds_mag_series,
        "vec_dirs_series": vec_dirs_series,
        "d_mag_dt_series": d_mag_dt_series,
    }


class AcquisitionPatchWindow(tk.Toplevel):
    """Controls + plots; mesh coloring is on the main 3D Carto pane."""

    def __init__(self, app: "App") -> None:
        super().__init__(app)
        self.app = app
        self.title("Acquisition patch analysis")
        self.transient(app)
        self.attributes("-topmost", True)
        self.geometry("720x660")

        self._snap_i = int(app.i)
        self._snap_j = int(app.j)
        self._snap_triple = bool(app.triple_active)
        self._snap_vt = bool(app.VT_active)

        self._viewer = _mesh_viewer(app)
        if self._viewer is None:
            messagebox.showerror(
                "3D viewer",
                "The Carto 3D mesh pane is not available.\n"
                "Start the app with the mesh viewer enabled, then open Patch analysis again.",
                parent=self,
            )
            self.after(0, self.destroy)
            return

        self._sec_idx: int = 0
        self._t: np.ndarray | None = None
        self._S: np.ndarray | None = None
        self._labels: list[str] = []
        self._ref_t_ms: np.ndarray | None = None
        self._ref_y: np.ndarray | None = None
        self._ref_m1: np.ndarray | None = None
        self._ref_m2: np.ndarray | None = None
        self._ref_name = "reference"
        self._ref_vline = None
        self._L = None
        self._graph = None
        self._V: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._anchor_idx: np.ndarray | None = None
        self._dSdt: np.ndarray | None = None
        self._local_r: float = 1.0
        self._patch_R: float = 1.0
        self._dvds_mag_series: np.ndarray | None = None
        self._vec_dirs_series: np.ndarray | None = None
        self._d_mag_dt_series: np.ndarray | None = None
        self._compute_mask: np.ndarray | None = None
        self._interest_spans_ms: list[tuple[float, float]] = []
        self._ref_drag_active = False
        self._updating_time = False
        self._patch_on_viewer = False
        self._prev_pick_cb = None
        self._selected_electrode_i: int | None = None
        self._display_mode = tk.StringVar(value="v")
        self._cb_amp: tuple[float, float] = (0.0, 1.0)
        self._cb_dv: tuple[float, float] = (0.0, 1.0)
        self._cb_dvds: tuple[float, float] = (0.0, 1.0)
        self._cb_ddvds_dt: tuple[float, float] = (0.0, 1.0)
        self._cb_amp_global: tuple[float, float] = (0.0, 1.0)
        self._cb_dv_global: tuple[float, float] = (0.0, 1.0)
        self._cb_dvds_global: tuple[float, float] = (0.0, 1.0)
        self._cb_ddvds_dt_global: tuple[float, float] = (0.0, 1.0)
        self._build_busy = False
        self._build_queue: queue.Queue = queue.Queue()
        self._build_thread: threading.Thread | None = None
        self._patch_ready = False

        body = ttk.Frame(self, padding=8)
        body.pack(fill="both", expand=True)

        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True)

        left_outer = ttk.Frame(body)
        left_outer.pack(side="left", fill="y", padx=(0, 10))
        self._left_canvas = tk.Canvas(left_outer, highlightthickness=0, width=300)
        left_vsb = ttk.Scrollbar(left_outer, orient="vertical", command=self._left_canvas.yview)
        self._left_canvas.configure(yscrollcommand=left_vsb.set)
        left_vsb.pack(side="right", fill="y")
        self._left_canvas.pack(side="left", fill="y")
        left = ttk.Frame(self._left_canvas)
        self._left_canvas_window = self._left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _left_configure(_evt=None) -> None:
            self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all"))
            try:
                self._left_canvas.itemconfigure(self._left_canvas_window, width=self._left_canvas.winfo_width())
            except tk.TclError:
                pass

        left.bind("<Configure>", _left_configure)

        def _left_wheel(evt) -> None:
            try:
                self._left_canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")
            except tk.TclError:
                pass

        def _bind_wheel(_evt) -> None:
            self._left_canvas.bind_all("<MouseWheel>", _left_wheel)

        def _unbind_wheel(_evt) -> None:
            self._left_canvas.unbind_all("<MouseWheel>")

        self._left_canvas.bind("<Enter>", _bind_wheel)
        self._left_canvas.bind("<Leave>", _unbind_wheel)

        ttk.Label(left, text="Acquisitions (Left/Right in main app)").pack(anchor="w")
        self._list = tk.Listbox(left, width=38, height=10, exportselection=False)
        self._list.pack(fill="y")
        self._fill_acquisition_list()
        self._list.bind("<<ListboxSelect>>", self._on_select_acquisition)

        self._acq_info = ttk.Label(left, text="", wraplength=260, foreground="#444")
        self._acq_info.pack(anchor="w", pady=(6, 0))

        ttk.Label(left, text="Geodesic patch radius (mesh units)").pack(anchor="w", pady=(8, 0))
        self._r_var = tk.DoubleVar(value=0.0)
        self._r_spin = ttk.Spinbox(
            left, from_=0.0, to=1e6, increment=0.5, width=12, textvariable=self._r_var
        )
        self._r_spin.pack(anchor="w")
        ttk.Label(left, text="0 = auto (15× mean edge)").pack(anchor="w")

        self._build_btn = ttk.Button(left, text="Build patch", command=self._on_build)
        self._build_btn.pack(pady=8, anchor="w")

        disp_fr = ttk.LabelFrame(left, text="Mesh display", padding=4)
        disp_fr.pack(fill="x", anchor="w", pady=(6, 0))
        for val, txt in (
            ("v", "V (mV)"),
            ("dvdt", "dV/dt (mV/ms)"),
            ("dvds", "dV/ds + vectors"),
            ("d_dvds_dt", "d(dV/ds)/dt + vectors"),
        ):
            ttk.Radiobutton(
                disp_fr, text=txt, value=val, variable=self._display_mode, command=self._on_display_mode
            ).pack(anchor="w")

        ttk.Label(left, text="Vector arrow scale (mesh units)").pack(anchor="w", pady=(6, 0))
        self._vec_scale_var = tk.DoubleVar(value=0.0)
        self._vec_scale_spin = ttk.Spinbox(
            left, from_=0.0, to=1e6, increment=0.05, width=10, textvariable=self._vec_scale_var
        )
        self._vec_scale_spin.pack(anchor="w")
        ttk.Label(left, text="0 = auto from patch radius").pack(anchor="w")

        self._cb_info = ttk.Label(left, text="Color bar: build patch first", wraplength=260, foreground="#555")
        self._cb_info.pack(anchor="w", pady=(4, 0))

        cb_lim = ttk.LabelFrame(left, text="Color limits (active mode)", padding=4)
        cb_lim.pack(fill="x", anchor="w", pady=(6, 0))

        self._lim_vmin_var = tk.DoubleVar(value=0.0)
        self._lim_vmax_var = tk.DoubleVar(value=1.0)
        lim_row = ttk.Frame(cb_lim)
        lim_row.pack(fill="x", anchor="w", pady=1)
        self._lim_mode_lbl = ttk.Label(lim_row, text="V (mV)", width=14)
        self._lim_mode_lbl.pack(side="left")
        self._lim_vmin_spin = ttk.Spinbox(
            lim_row, from_=-1e9, to=1e9, increment=0.01, width=9, textvariable=self._lim_vmin_var, format="%.6g"
        )
        self._lim_vmin_spin.pack(side="left", padx=(0, 4))
        self._lim_vmax_spin = ttk.Spinbox(
            lim_row, from_=-1e9, to=1e9, increment=0.01, width=9, textvariable=self._lim_vmax_var, format="%.6g"
        )
        self._lim_vmax_spin.pack(side="left")
        cb_btn = ttk.Frame(cb_lim)
        cb_btn.pack(fill="x", anchor="w", pady=(4, 0))
        self._cb_apply_btn = ttk.Button(cb_btn, text="Apply limits", command=self._on_apply_color_limits)
        self._cb_apply_btn.pack(side="left")
        self._cb_reset_btn = ttk.Button(cb_btn, text="Reset global", command=self._on_reset_color_limits)
        self._cb_reset_btn.pack(side="left", padx=(6, 0))

        cmap_fr = ttk.LabelFrame(left, text="Colormap style", padding=4)
        cmap_fr.pack(fill="x", anchor="w", pady=(6, 0))
        cr = ttk.Frame(cmap_fr)
        cr.pack(fill="x")
        ttk.Label(cr, text="Map").pack(side="left")
        self._cmap_var = tk.StringVar(value="turbo")
        self._cmap_combo = ttk.Combobox(
            cr, textvariable=self._cmap_var, values=list(cm.COLORMAPS), width=16, state="readonly"
        )
        self._cmap_combo.pack(side="left", padx=4)
        self._cmap_rev_var = tk.IntVar(value=0)
        self._cmap_rev_chk = ttk.Checkbutton(cmap_fr, text="Reverse", variable=self._cmap_rev_var)
        self._cmap_rev_chk.pack(anchor="w")
        br = ttk.Frame(cmap_fr)
        br.pack(fill="x", pady=(2, 0))
        ttk.Label(br, text="Bins").pack(side="left")
        self._cmap_bins_var = tk.IntVar(value=256)
        self._cmap_bins_spin = ttk.Spinbox(br, from_=2, to=256, width=6, textvariable=self._cmap_bins_var)
        self._cmap_bins_spin.pack(side="left", padx=4)
        self._cmap_apply_btn = ttk.Button(cmap_fr, text="Apply colormap", command=self._on_apply_colormap)
        self._cmap_apply_btn.pack(anchor="w", pady=(4, 0))

        ttk.Label(left, text="Time").pack(anchor="w", pady=(8, 0))
        prog_fr = ttk.Frame(left)
        prog_fr.pack(fill="x", anchor="w")
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress = ttk.Progressbar(
            prog_fr, orient="horizontal", length=250, mode="determinate", variable=self._progress_var
        )
        self._progress.pack(fill="x")
        self._progress_lbl = ttk.Label(prog_fr, text="Idle", foreground="#555")
        self._progress_lbl.pack(anchor="w")

        self._time_var = tk.IntVar(value=0)
        self._scale = ttk.Scale(
            left,
            from_=0,
            to=1,
            orient="horizontal",
            length=250,
            variable=self._time_var,
            command=self._on_slider,
        )
        self._scale.pack(anchor="w", fill="x")
        self._time_lbl = ttk.Label(left, text="—")
        self._time_lbl.pack(anchor="w")

        pt_fr = ttk.LabelFrame(left, text="Selected patch point (click point on 3D mesh)", padding=4)
        pt_fr.pack(fill="x", anchor="w", pady=(8, 0))
        self._point_info = ttk.Label(pt_fr, text="—", wraplength=280, justify="left", foreground="#222")
        self._point_info.pack(anchor="w")

        ttk.Label(
            left,
            text="dV/ds: strongest |ΔV|/Δs on shortest paths from each electrode.\n"
            "Vectors scale with |dV/ds|; arrow scale sets mesh units per magnitude.",
            foreground="#555",
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

        self._fig = Figure(figsize=(6.2, 6.8), dpi=100)
        gs = self._fig.add_gridspec(4, 1, height_ratios=[2.2, 1.0, 1.0, 1.0])
        self._ax_ref = self._fig.add_subplot(gs[0])
        self._ax_pt_v = self._fig.add_subplot(gs[1])
        self._ax_pt_dvdt = self._fig.add_subplot(gs[2])
        self._ax_pt_dvds = self._fig.add_subplot(gs[3])
        self._fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.08, hspace=0.55)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas.mpl_connect("button_press_event", self._on_ref_canvas_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_ref_canvas_motion)
        self._canvas.mpl_connect("button_release_event", self._on_ref_canvas_release)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Destroy>", self._on_destroy)

        if not self._viewer.begin_patch_preview(self._patch_field_label()):
            messagebox.showwarning(
                "Mesh",
                "Could not load mesh on the 3D viewer. Build patch after the mesh is ready.",
                parent=self,
            )
        else:
            self._patch_on_viewer = True
            self._install_mesh_pick()

        self.after(80, lambda: self.attributes("-topmost", False))
        if self.app.carto.cont:
            self._list.selection_set(0)
            self._on_select_acquisition()

    def _install_mesh_pick(self) -> None:
        if self._viewer is None:
            return
        self._prev_pick_cb = getattr(self._viewer, "on_pick_callback", None)
        self._viewer.on_pick_callback = self._on_mesh_pick

    def _restore_mesh_pick(self) -> None:
        if self._viewer is None:
            return
        try:
            self._viewer.on_pick_callback = self._prev_pick_cb
        except Exception:
            pass
        self._prev_pick_cb = None

    def _on_mesh_pick(self, kind: str, payload, info: dict) -> None:
        if kind == "patch_anchor" and payload is not None:
            self._select_patch_point(int(payload))
            return
        if callable(self._prev_pick_cb):
            try:
                self._prev_pick_cb(kind, payload, info)
            except Exception:
                traceback.print_exc()

    def _electrode_metrics_at(self, ei: int, k: int) -> dict[str, float]:
        ei = int(ei)
        k = int(k)
        out = {"v": float("nan"), "dvdt": float("nan"), "dvds": float("nan"), "ddvds_dt": float("nan")}
        if self._S is None or self._dSdt is None or ei < 0 or ei >= self._S.shape[1]:
            return out
        k = int(np.clip(k, 0, self._S.shape[0] - 1))
        out["v"] = float(self._S[k, ei])
        out["dvdt"] = float(self._dSdt[k, ei])
        if self._dvds_mag_series is not None and k < self._dvds_mag_series.shape[0]:
            m = float(self._dvds_mag_series[k, ei])
            if np.isfinite(m):
                out["dvds"] = abs(m)
        if (
            not np.isfinite(out["dvds"])
            and self._dvds_mag_series is not None
            and k < self._dvds_mag_series.shape[0]
            and ei < self._dvds_mag_series.shape[1]
        ):
            m = float(self._dvds_mag_series[k, ei])
            if np.isfinite(m):
                out["dvds"] = abs(m)
        if self._d_mag_dt_series is not None and k < self._d_mag_dt_series.shape[0]:
            m = float(self._d_mag_dt_series[k, ei])
            if np.isfinite(m):
                out["ddvds_dt"] = abs(m)
        return out

    def _update_point_info(self, ei: int | None, k: int) -> None:
        if ei is None or not self._patch_ready:
            self._point_info.config(text="Click a patch electrode point on the 3D mesh.")
            return
        label = self._labels[ei] if ei < len(self._labels) else f"#{ei}"
        m = self._electrode_metrics_at(ei, k)
        t_ms = self._time_ms_at(k)
        t_txt = f"{t_ms:.2f} ms" if t_ms is not None else f"sample {k}"
        self._point_info.config(
            text=(
                f"{label}  @  {t_txt}\n"
                f"V = {m['v']:.4g} mV\n"
                f"dV/dt = {m['dvdt']:.4g} mV/ms\n"
                f"|dV/ds| = {m['dvds']:.4g}\n"
                f"|d(dV/ds)/dt| = {m['ddvds_dt']:.4g}"
            )
        )

    def _select_patch_point(self, ei: int) -> None:
        self._selected_electrode_i = int(ei)
        if self._viewer is not None and self._patch_on_viewer:
            try:
                self._viewer.set_patch_preview_selected_anchor(int(ei))
            except Exception:
                traceback.print_exc()
        try:
            k = int(self._time_var.get())
        except Exception:
            k = 0
        self._update_point_info(int(ei), k)
        self._draw_electrode_timeseries(int(ei), k)
        if self._patch_ready:
            self._push_vectors_to_viewer(k)

    def _display_mode_key(self) -> str:
        return str(self._display_mode.get() or "v")

    def _patch_field_label(self) -> str:
        mode = self._display_mode_key()
        labels = {
            "v": f"patch:acq{self._sec_idx} V (mV)",
            "dvdt": f"patch:acq{self._sec_idx} dV/dt (mV/ms)",
            "dvds": f"patch:acq{self._sec_idx} dV/ds",
            "d_dvds_dt": f"patch:acq{self._sec_idx} d(dV/ds)/dt",
        }
        return labels.get(mode, labels["v"])

    def _active_cb_limits(self) -> tuple[tuple[float, float], str]:
        mode = self._display_mode_key()
        if mode == "dvdt":
            return self._cb_dv, "dV/dt (mV/ms)"
        if mode == "dvds":
            return self._cb_dvds, "dV/ds"
        if mode == "d_dvds_dt":
            return self._cb_ddvds_dt, "d(dV/ds)/dt"
        return self._cb_amp, "V (mV)"

    def _active_cb_global(self) -> tuple[float, float]:
        mode = self._display_mode_key()
        if mode == "dvdt":
            return self._cb_dv_global
        if mode == "dvds":
            return self._cb_dvds_global
        if mode == "d_dvds_dt":
            return self._cb_ddvds_dt_global
        return self._cb_amp_global

    def _set_active_cb_limits(self, lim: tuple[float, float]) -> None:
        mode = self._display_mode_key()
        if mode == "dvdt":
            self._cb_dv = lim
        elif mode == "dvds":
            self._cb_dvds = lim
        elif mode == "d_dvds_dt":
            self._cb_ddvds_dt = lim
        else:
            self._cb_amp = lim

    def _fill_acquisition_list(self) -> None:
        self._list.delete(0, tk.END)
        cont = getattr(self.app.carto, "cont", None) or []
        for i, sec in enumerate(cont):
            try:
                n = len(sec[0])
                name = str(sec[1])
            except Exception:
                n, name = 0, "?"
            self._list.insert(tk.END, f"{i}: {name} — {n} points")

    def _set_build_busy(self, busy: bool) -> None:
        self._build_busy = bool(busy)
        state = tk.DISABLED if busy else tk.NORMAL
        try:
            self.config(cursor="watch" if busy else "")
            self.app.config(cursor="watch" if busy else "")
        except Exception:
            pass
        for w in (
            self._list,
            self._build_btn,
            self._r_spin,
            self._vec_scale_spin,
            self._scale,
            self._lim_vmin_spin,
            self._lim_vmax_spin,
            self._cb_apply_btn,
            self._cb_reset_btn,
            self._cmap_combo,
            self._cmap_rev_chk,
            self._cmap_bins_spin,
            self._cmap_apply_btn,
        ):
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    def _poll_build_queue(self) -> None:
        try:
            while True:
                msg = self._build_queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, cur, total, text = msg
                    if total > 0:
                        self._progress_var.set(100.0 * float(cur) / float(total))
                    self._progress_lbl.config(text=f"{text}  ({cur}/{total})")
                elif kind == "done":
                    self._on_build_done(msg[1])
                    return
                elif kind == "error":
                    self._on_build_failed(msg[1])
                    return
        except queue.Empty:
            pass
        if self._build_busy:
            self.after(50, self._poll_build_queue)

    def _on_select_acquisition(self, _evt=None) -> None:
        if self._build_busy:
            return
        sel = self._list.curselection()
        if not sel:
            return
        self._sec_idx = int(sel[0])
        self._patch_ready = False
        self._S = None
        self._dSdt = None
        self._ref_t_ms, self._ref_y, self._ref_m1, self._ref_m2, self._ref_name = _reference_traces(
            self.app, self._sec_idx
        )
        self._update_interest_windows()
        try:
            self._t, self._S, self._labels, pts = _acquisition_stack(self.app, self._sec_idx)
            self._update_interest_windows()
            n_df = len(self.app.carto.cont[self._sec_idx][0])
            n_t = int(self._t.size) if self._t is not None else 0
            self._acq_info.config(
                text=f"Acq {self._sec_idx}: {pts.shape[0]}/{n_df} pts with unipolar, {n_t} samples.\n"
                "Build patch (runs in background with progress above slider)."
            )
            if n_t > 0:
                self._scale.configure(from_=0, to=max(0, n_t - 1))
                self._time_var.set(0)
        except Exception:
            traceback.print_exc()
        self._draw_reference(0)
        self._selected_electrode_i = None
        self._update_point_info(None, 0)
        self._draw_electrode_timeseries(None, 0)
        if self._viewer is not None and self._patch_on_viewer:
            self._viewer.set_patch_preview_label(self._patch_field_label())
            self._viewer.set_patch_preview_anchors(None)
            n = int(np.asarray(self.app.carto.vertices).shape[0])
            self._viewer.set_patch_preview_field(np.full(n, np.nan, dtype=np.float32))

    def _mean_edge_hint(self) -> float:
        V = getattr(self.app.carto, "vertices", None)
        F = getattr(self.app.carto, "triangles", None)
        if V is None or F is None or np.asarray(V).size == 0:
            return 1.0
        try:
            return float(
                lap.mean_edge_length(np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64))
            )
        except Exception:
            return 1.0

    def _ensure_mesh_loaded(self) -> bool:
        carto = self.app.carto
        if getattr(carto, "vertices", None) is not None and np.asarray(carto.vertices).size > 0:
            return True
        try:
            carto.pars_mesh_file_with_electrode()
        except Exception:
            traceback.print_exc()
            return False
        if self._viewer is not None:
            try:
                self._viewer._load_mesh()
            except Exception:
                traceback.print_exc()
        return getattr(carto, "vertices", None) is not None and np.asarray(carto.vertices).size > 0

    def _on_build(self) -> None:
        if self._viewer is None or self._build_busy:
            return
        if not self._ensure_mesh_loaded():
            messagebox.showerror(
                "Mesh missing",
                "Could not load the anatomical mesh (.mesh in the Carto export folder).",
                parent=self,
            )
            return
        if not self._patch_on_viewer:
            if self._viewer.begin_patch_preview(self._patch_field_label()):
                self._patch_on_viewer = True

        carto = self.app.carto
        V = np.asarray(carto.vertices, dtype=np.float64).reshape(-1, 3)
        F = np.asarray(carto.triangles, dtype=np.int64).reshape(-1, 3)

        t, S, labels, pts = _acquisition_stack(self.app, self._sec_idx)
        if S.size == 0 or t.size == 0:
            messagebox.showerror("Signals", "No unipolar columns resolved for this acquisition.", parent=self)
            return
        if pts.shape[0] != S.shape[1]:
            messagebox.showerror("Internal", "Position/trace alignment failed.", parent=self)
            return

        self._ref_t_ms, self._ref_y, self._ref_m1, self._ref_m2, self._ref_name = _reference_traces(
            self.app, self._sec_idx
        )

        r_user = float(self._r_var.get())
        h = self._mean_edge_hint()
        R = 15.0 * h if (not np.isfinite(r_user) or r_user <= 0) else r_user

        try:
            anc = lap.map_points_to_vertices(V, pts)
            graph = lap.build_mesh_graph(V, F)
            mask = lap.dijkstra_radius_mask(graph, anc, R)
            L = lap.cot_laplacian(V, F)
            local_r = max(3.0 * h, 0.35 * R)
        except Exception:
            traceback.print_exc()
            messagebox.showerror("Patch", "Failed to build graph / Laplacian.", parent=self)
            return

        if not mask.any():
            messagebox.showerror("Patch", "Empty patch — increase radius.", parent=self)
            return

        dSdt = _compute_dvdt_mv_per_ms(S, t)
        self._update_interest_windows()
        sample_mask = self._compute_mask
        if sample_mask is None or sample_mask.size != S.shape[0]:
            sample_mask = np.ones(S.shape[0], dtype=bool)

        payload = {
            "L": L,
            "graph": graph,
            "V": V,
            "anc": anc,
            "mask": mask,
            "S": S,
            "dSdt": dSdt,
            "t": t,
            "labels": labels,
            "R": R,
            "local_r": local_r,
            "n_patch": int(mask.sum()),
            "sample_mask": sample_mask,
        }

        self._set_build_busy(True)
        self._patch_ready = False
        self._progress_var.set(0.0)
        self._progress_lbl.config(text="Starting…")
        self._cb_info.config(text="Computing global limits in background…")

        q = self._build_queue

        def worker() -> None:
            try:

                def prog(k: int, n: int, text: str) -> None:
                    q.put(("progress", k, n, text))

                ranges = _global_ranges_worker(
                    payload["L"],
                    payload["graph"],
                    payload["V"],
                    payload["anc"],
                    payload["mask"],
                    payload["S"],
                    payload["dSdt"],
                    payload["t"],
                    payload["local_r"],
                    sample_mask=payload["sample_mask"],
                    progress=prog,
                )
                q.put(("done", {**payload, **ranges}))
            except Exception as exc:
                traceback.print_exc()
                q.put(("error", exc))

        self._build_thread = threading.Thread(target=worker, daemon=True, name="patch-global-range")
        self._build_thread.start()
        self.after(50, self._poll_build_queue)

    def _on_build_done(self, payload: dict) -> None:
        self._set_build_busy(False)
        self._progress_var.set(100.0)
        self._progress_lbl.config(text="Done")

        self._L = payload["L"]
        self._graph = payload["graph"]
        self._V = payload["V"]
        self._mask = payload["mask"]
        self._anchor_idx = payload["anc"]
        self._t = payload["t"]
        self._S = payload["S"]
        self._labels = payload["labels"]
        self._dSdt = payload["dSdt"]
        self._local_r = float(payload["local_r"])
        self._patch_R = float(payload["R"])
        self._dvds_mag_series = payload["dvds_mag_series"]
        self._vec_dirs_series = payload.get("vec_dirs_series")
        self._d_mag_dt_series = payload["d_mag_dt_series"]
        self._compute_mask = payload.get("sample_mask")
        self._update_interest_windows()
        self._cb_amp_global = payload["cb_amp"]
        self._cb_dv_global = payload["cb_dv"]
        self._cb_dvds_global = payload["cb_dvds"]
        self._cb_ddvds_dt_global = payload["cb_ddvds_dt"]
        self._cb_amp = self._cb_amp_global
        self._cb_dv = self._cb_dv_global
        self._cb_dvds = self._cb_dvds_global
        self._cb_ddvds_dt = self._cb_ddvds_dt_global
        self._patch_ready = True
        if float(self._vec_scale_var.get()) <= 0:
            self._vec_scale_var.set(max(0.05, 0.25 * self._local_r))
        self._sync_color_limit_vars_from_global()
        self._update_cb_info_label()
        self._on_apply_colormap()
        if self._viewer is not None and self._patch_on_viewer and self._anchor_idx is not None:
            try:
                self._viewer.set_patch_preview_anchors(
                    self._anchor_idx,
                    self._labels,
                    selected=self._selected_electrode_i,
                )
            except Exception:
                traceback.print_exc()

        n = int(self._S.shape[0])
        n_comp = int(np.sum(self._compute_mask)) if self._compute_mask is not None else n
        self._scale.configure(from_=0, to=max(0, n - 1))
        first = self._first_compute_index()
        self._apply_time_index(first if first is not None else 0)
        messagebox.showinfo(
            "Patch",
            f"Ready on main 3D mesh: R={payload['R']:.4g}, "
            f"{payload['n_patch']} patch vertices, {n_comp}/{n} samples in windows of interest.",
            parent=self,
        )

    def _on_build_failed(self, exc: BaseException) -> None:
        self._set_build_busy(False)
        self._progress_var.set(0.0)
        self._progress_lbl.config(text="Failed")
        messagebox.showerror("Patch build failed", str(exc), parent=self)

    def _phi_v_at_time(self, k: int) -> np.ndarray | None:
        if self._S is None or self._L is None or self._anchor_idx is None or self._mask is None:
            return None
        k = int(np.clip(k, 0, self._S.shape[0] - 1))
        vals = np.asarray(self._S[k, :], dtype=np.float64).ravel()
        return lap.harmonic_interpolate(self._L, self._anchor_idx, vals, free_mask=self._mask)

    def _phi_at_time(self, k: int) -> np.ndarray | None:
        if not self._patch_ready or self._L is None or self._anchor_idx is None or self._mask is None:
            return None
        mode = self._display_mode_key()
        k = int(np.clip(k, 0, self._S.shape[0] - 1)) if self._S is not None else 0
        if mode == "dvdt":
            if self._dSdt is None:
                return None
            vals = np.asarray(self._dSdt[k, :], dtype=np.float64).ravel()
            return lap.harmonic_interpolate(self._L, self._anchor_idx, vals, free_mask=self._mask)
        if mode == "dvds":
            if self._dvds_mag_series is not None and k < self._dvds_mag_series.shape[0]:
                row = np.abs(np.nan_to_num(self._dvds_mag_series[k, :], nan=0.0))
                return lap.harmonic_interpolate(self._L, self._anchor_idx, row, free_mask=self._mask)
            return None
        if mode == "d_dvds_dt":
            if self._d_mag_dt_series is None:
                return None
            row = np.abs(np.nan_to_num(self._d_mag_dt_series[k, :], nan=0.0))
            return lap.harmonic_interpolate(self._L, self._anchor_idx, row, free_mask=self._mask)
        return self._phi_v_at_time(k)

    def _vector_scale(self) -> float:
        try:
            s = float(self._vec_scale_var.get())
        except tk.TclError:
            s = 0.0
        if s > 0:
            return s
        return max(0.05, 0.25 * float(self._local_r))

    def _update_interest_windows(self) -> None:
        if self._ref_y is None or np.asarray(self._ref_y).size == 0:
            self._compute_mask = None
            self._interest_spans_ms = []
            return
        if self._t is not None and self._t.size > 0:
            t_s = self._t
        elif self._ref_t_ms is not None and self._ref_t_ms.size > 0:
            t_s = np.asarray(self._ref_t_ms, dtype=np.float64) / 1000.0
        else:
            self._compute_mask = None
            self._interest_spans_ms = []
            return
        mask, spans = _interest_sample_mask(t_s, self._ref_y, stim_pad_ms=100.0, sr_pad_ms=50.0, fs=1000.0)
        self._compute_mask = mask
        self._interest_spans_ms = spans

    def _in_compute_window(self, k: int) -> bool:
        if self._compute_mask is None or self._compute_mask.size == 0:
            return True
        k = int(np.clip(k, 0, self._compute_mask.size - 1))
        return bool(self._compute_mask[k])

    def _index_from_ms(self, t_ms: float) -> int:
        n = 0
        if self._t is not None and self._t.size > 0:
            t_ms_arr = np.asarray(self._t, dtype=np.float64).ravel() * 1000.0
            n = int(t_ms_arr.size)
            return int(np.clip(np.argmin(np.abs(t_ms_arr - float(t_ms))), 0, max(0, n - 1)))
        if self._ref_t_ms is not None and self._ref_t_ms.size > 0:
            n = int(self._ref_t_ms.size)
            return int(np.clip(np.argmin(np.abs(self._ref_t_ms - float(t_ms))), 0, max(0, n - 1)))
        return 0

    def _set_time_index(self, k: int) -> None:
        """Move slider + reference cursor; update mesh only inside interest windows."""
        if self._updating_time:
            return
        self._updating_time = True
        try:
            self._set_time_index_impl(k)
        finally:
            self._updating_time = False

    def _set_time_index_impl(self, k: int) -> None:
        if self._S is not None and self._S.size > 0:
            k = int(np.clip(k, 0, self._S.shape[0] - 1))
        else:
            k = int(max(0, k))
        try:
            self._time_var.set(k)
        except tk.TclError:
            pass
        in_win = self._in_compute_window(k)
        t_txt = ""
        t_ms = self._time_ms_at(k)
        if t_ms is not None:
            t_txt = f"  t={t_ms:.2f} ms"
        mode_labels = {
            "v": "V (mV)",
            "dvdt": "dV/dt (mV/ms)",
            "dvds": "dV/ds + vectors",
            "d_dvds_dt": "d(dV/ds)/dt + vectors",
        }
        mode = mode_labels.get(self._display_mode_key(), "V")
        win_txt = "computed window" if in_win else "ignored (outside windows of interest)"
        self._time_lbl.config(text=f"sample {k}{t_txt}  —  {mode}  [{win_txt}]")

        self._draw_reference(k)

        if self._selected_electrode_i is not None:
            self._update_point_info(self._selected_electrode_i, k)

        if self._patch_ready and self._S is not None and self._dSdt is not None and in_win:
            self._push_phi_to_viewer(k)
        elif self._patch_ready and self._viewer is not None and self._patch_on_viewer:
            try:
                n = int(np.asarray(self.app.carto.vertices).shape[0])
                self._viewer.clear_patch_preview_vectors()
                self._viewer.set_patch_preview_field(np.full(n, np.nan, dtype=np.float32))
            except Exception:
                traceback.print_exc()

        self._draw_electrode_timeseries(self._selected_electrode_i, k)
        self._canvas.draw_idle()

    def _electrode_time_ms(self) -> np.ndarray | None:
        if self._S is None:
            return None
        n = int(self._S.shape[0])
        if self._t is not None and self._t.size >= n:
            return np.asarray(self._t[:n], dtype=np.float64).ravel() * 1000.0
        if self._ref_t_ms is not None and self._ref_t_ms.size >= n:
            return np.asarray(self._ref_t_ms[:n], dtype=np.float64).ravel()
        return np.arange(n, dtype=np.float64)

    def _shade_interest_on_ax(self, ax, ylo: float, yhi: float, t_ms: np.ndarray | None = None) -> None:
        if t_ms is None:
            t_ms = self._electrode_time_ms()
        if t_ms is None:
            return
        t_ms = np.asarray(t_ms, dtype=np.float64).ravel()
        if self._compute_mask is not None:
            n = min(t_ms.size, self._compute_mask.size)
            if n > 0:
                i = 0
                while i < n:
                    if self._compute_mask[i]:
                        i += 1
                        continue
                    j = i + 1
                    while j < n and not self._compute_mask[j]:
                        j += 1
                    ax.add_patch(
                        Rectangle(
                            (float(t_ms[i]), ylo),
                            float(t_ms[j - 1] - t_ms[i]) if j > i + 1 else 0.001,
                            yhi - ylo,
                            facecolor="#555555",
                            alpha=0.18,
                            edgecolor="none",
                            zorder=0,
                        )
                    )
                    i = j
        for t0, t1 in self._interest_spans_ms:
            ax.axvspan(t0, t1, color="#3d8c40", alpha=0.12, zorder=1)

    def _draw_electrode_timeseries(self, ei: int | None, k: int) -> None:
        for ax in (self._ax_pt_v, self._ax_pt_dvdt, self._ax_pt_dvds):
            ax.clear()

        if ei is None or not self._patch_ready or self._S is None or self._dSdt is None:
            self._ax_pt_v.set_title("Select a patch electrode on the 3D mesh")
            self._ax_pt_v.axis("off")
            self._ax_pt_dvdt.axis("off")
            self._ax_pt_dvds.axis("off")
            return

        ei = int(ei)
        if ei < 0 or ei >= self._S.shape[1]:
            self._ax_pt_v.set_title("Invalid electrode selection")
            self._ax_pt_v.axis("off")
            self._ax_pt_dvdt.axis("off")
            self._ax_pt_dvds.axis("off")
            return

        t_ms = self._electrode_time_ms()
        if t_ms is None:
            return
        n = min(int(t_ms.size), int(self._S.shape[0]))
        t_ms = t_ms[:n]
        label = self._labels[ei] if ei < len(self._labels) else f"#{ei}"
        tx_ms = self._time_ms_at(k)

        def _plot_trace(ax, y: np.ndarray, ylabel: str, title: str, color: str, *, xlabel: bool = False) -> None:
            y = np.asarray(y, dtype=np.float64).ravel()[:n]
            ylo, yhi = float(np.nanmin(y)), float(np.nanmax(y))
            if not np.isfinite(ylo) or not np.isfinite(yhi):
                ylo, yhi = -1.0, 1.0
            if yhi <= ylo:
                yhi = ylo + 1.0
            pad = 0.08 * (yhi - ylo)
            ylo -= pad
            yhi += pad
            self._shade_interest_on_ax(ax, ylo, yhi, t_ms)
            xm, ym = _decimate_xy(t_ms, y)
            ax.plot(xm, ym, color=color, linewidth=0.9, zorder=3)
            if tx_ms is not None:
                ax.axvline(tx_ms, color="orangered", linewidth=1.2, linestyle="--", zorder=5)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))
            if xlabel:
                ax.set_xlabel("time (ms)")

        v = np.asarray(self._S[:n, ei], dtype=np.float64)
        dv = np.asarray(self._dSdt[:n, ei], dtype=np.float64)
        if self._dvds_mag_series is not None and self._dvds_mag_series.shape[0] >= n:
            dvds = np.abs(np.asarray(self._dvds_mag_series[:n, ei], dtype=np.float64))
        else:
            dvds = np.full(n, np.nan, dtype=np.float64)

        _plot_trace(self._ax_pt_v, v, "mV", f"{label} — V", "#2a6fdb")
        _plot_trace(self._ax_pt_dvdt, dv, "mV/ms", f"{label} — dV/dt", "#1a9e6e")
        _plot_trace(self._ax_pt_dvds, dvds, "|dV/ds|", f"{label} — |dV/ds|", "#7b4fd4", xlabel=True)

    def _on_ref_canvas_press(self, event) -> None:
        if self._build_busy or event.inaxes is not self._ax_ref or event.xdata is None:
            return
        self._ref_drag_active = True
        self._set_time_index(self._index_from_ms(float(event.xdata)))

    def _on_ref_canvas_motion(self, event) -> None:
        if not self._ref_drag_active or event.inaxes is not self._ax_ref or event.xdata is None:
            return
        self._set_time_index(self._index_from_ms(float(event.xdata)))

    def _on_ref_canvas_release(self, _event) -> None:
        self._ref_drag_active = False

    def _first_compute_index(self) -> int | None:
        if self._compute_mask is None or not np.any(self._compute_mask):
            return None
        return int(np.argmax(self._compute_mask))

    def _push_vectors_to_viewer(self, k: int) -> None:
        if self._viewer is None or not self._patch_on_viewer:
            return
        mode = self._display_mode_key()
        if mode not in ("dvds", "d_dvds_dt"):
            self._viewer.clear_patch_preview_vectors()
            return
        if (
            not self._patch_ready
            or self._V is None
            or self._anchor_idx is None
            or self._vec_dirs_series is None
            or self._dvds_mag_series is None
        ):
            self._viewer.clear_patch_preview_vectors()
            return
        k = int(np.clip(k, 0, self._dvds_mag_series.shape[0] - 1))
        if not self._in_compute_window(k):
            self._viewer.clear_patch_preview_vectors()
            return
        anc = np.asarray(self._anchor_idx, dtype=np.int64).ravel()
        origins = np.asarray(self._V[anc], dtype=np.float64)
        dirs = np.asarray(self._vec_dirs_series[k, :, :], dtype=np.float64)
        if mode == "d_dvds_dt" and self._d_mag_dt_series is not None:
            mags = np.abs(np.asarray(self._d_mag_dt_series[k, :], dtype=np.float64))
        else:
            mags = np.abs(np.asarray(self._dvds_mag_series[k, :], dtype=np.float64))
        # Drop electrodes with no valid direction at this time.
        ok = np.all(np.isfinite(dirs), axis=1) & np.isfinite(mags) & (mags > 0)
        dn = np.linalg.norm(dirs, axis=1)
        ok &= dn > 1e-12
        if not np.any(ok):
            self._viewer.clear_patch_preview_vectors()
            return
        self._viewer.set_patch_preview_vectors(
            origins[ok],
            dirs[ok],
            mags[ok],
            scale=self._vector_scale(),
        )

    def _sync_color_limit_vars_from_global(self) -> None:
        lim, label = self._active_cb_limits()
        _ = label
        glob = self._active_cb_global()
        self._lim_vmin_var.set(float(glob[0]))
        self._lim_vmax_var.set(float(glob[1]))
        self._lim_mode_lbl.config(text=label)

    def _read_color_limits_from_ui(self) -> tuple[float, float]:
        return (float(self._lim_vmin_var.get()), float(self._lim_vmax_var.get()))

    def _update_cb_info_label(self) -> None:
        self._cb_info.config(
            text=(
                f"V [{self._cb_amp[0]:.3g},{self._cb_amp[1]:.3g}]  "
                f"dV/dt [{self._cb_dv[0]:.3g},{self._cb_dv[1]:.3g}]  "
                f"dV/ds [{self._cb_dvds[0]:.3g},{self._cb_dvds[1]:.3g}]  "
                f"d(dV/ds)/dt [{self._cb_ddvds_dt[0]:.3g},{self._cb_ddvds_dt[1]:.3g}]"
            )
        )

    def _on_apply_color_limits(self) -> None:
        if self._build_busy or not self._patch_ready:
            return
        try:
            lim = self._read_color_limits_from_ui()
        except tk.TclError:
            messagebox.showerror("Color limits", "Invalid min/max values.", parent=self)
            return
        if not np.isfinite(lim[0]) or not np.isfinite(lim[1]) or lim[0] >= lim[1]:
            messagebox.showerror("Color limits", "Need finite min < max.", parent=self)
            return
        self._set_active_cb_limits(lim)
        self._update_cb_info_label()
        try:
            k = int(self._time_var.get())
        except Exception:
            k = 0
        self._apply_time_index(k)

    def _on_reset_color_limits(self) -> None:
        if not self._patch_ready:
            return
        mode = self._display_mode_key()
        if mode == "dvdt":
            self._cb_dv = self._cb_dv_global
        elif mode == "dvds":
            self._cb_dvds = self._cb_dvds_global
        elif mode == "d_dvds_dt":
            self._cb_ddvds_dt = self._cb_ddvds_dt_global
        else:
            self._cb_amp = self._cb_amp_global
        self._sync_color_limit_vars_from_global()
        self._update_cb_info_label()
        try:
            k = int(self._time_var.get())
        except Exception:
            k = 0
        self._apply_time_index(k)

    def _on_apply_colormap(self) -> None:
        if self._viewer is None or not self._patch_on_viewer:
            return
        try:
            nb = int(self._cmap_bins_var.get())
            nb = max(2, min(256, nb))
        except tk.TclError:
            nb = 256
        name = str(self._cmap_var.get() or "turbo")
        if name not in cm.COLORMAPS:
            name = "turbo"
        try:
            self._viewer.set_patch_preview_color_style(
                cmap_name=name,
                reverse=bool(self._cmap_rev_var.get()),
                n_bins=nb,
                color_mode="standard",
            )
        except Exception:
            traceback.print_exc()

    def _apply_colorbar_to_viewer(self) -> None:
        if self._viewer is None or not self._patch_on_viewer or not self._patch_ready:
            return
        lim, _ = self._active_cb_limits()
        self._viewer.set_patch_preview_color_range(lim[0], lim[1], auto_range=False)
        self._viewer.set_patch_preview_label(self._patch_field_label())

    def _push_phi_to_viewer(self, k: int) -> None:
        if self._viewer is None or not self._patch_on_viewer or not self._patch_ready:
            return
        phi = self._phi_at_time(k)
        if phi is None:
            return
        try:
            self._viewer.set_patch_preview_label(self._patch_field_label())
            lim, _ = self._active_cb_limits()
            self._viewer.set_patch_preview_color_range(lim[0], lim[1], auto_range=False)
            self._viewer.set_patch_preview_field(phi)
            self._push_vectors_to_viewer(k)
        except Exception:
            traceback.print_exc()

    def _time_ms_at(self, k: int) -> float | None:
        if self._t is not None and self._t.size > 0:
            k = int(np.clip(k, 0, self._t.size - 1))
            return float(self._t[k]) * 1000.0
        if self._ref_t_ms is not None and self._ref_t_ms.size > 0:
            k = int(np.clip(k, 0, self._ref_t_ms.size - 1))
            return float(self._ref_t_ms[k])
        return None

    def _draw_reference(self, k: int) -> None:
        self._ax_ref.clear()
        self._ref_vline = None
        if self._ref_t_ms is not None and self._ref_y is not None and self._ref_t_ms.size > 0:
            t_ms = np.asarray(self._ref_t_ms, dtype=np.float64).ravel()
            y = np.asarray(self._ref_y, dtype=np.float64).ravel()
            n = min(t_ms.size, y.size)
            t_ms = t_ms[:n]
            y = y[:n]

            ylo, yhi = float(np.nanmin(y)), float(np.nanmax(y))
            if yhi <= ylo:
                yhi = ylo + 1.0
            pad = 0.06 * (yhi - ylo)
            ylo -= pad
            yhi += pad

            # Ignored regions (outside windows of interest).
            if self._compute_mask is not None and self._compute_mask.size == n:
                i = 0
                while i < n:
                    if self._compute_mask[i]:
                        i += 1
                        continue
                    j = i + 1
                    while j < n and not self._compute_mask[j]:
                        j += 1
                    self._ax_ref.add_patch(
                        Rectangle(
                            (float(t_ms[i]), ylo),
                            float(t_ms[j - 1] - t_ms[i]) if j > i + 1 else 0.001,
                            yhi - ylo,
                            facecolor="#555555",
                            alpha=0.22,
                            edgecolor="none",
                            zorder=0,
                        )
                    )
                    i = j
            for t0, t1 in self._interest_spans_ms:
                self._ax_ref.axvspan(t0, t1, color="#3d8c40", alpha=0.16, zorder=1)

            # Dim full trace; highlight computed segments.
            xm, ym = _decimate_xy(t_ms, y)
            self._ax_ref.plot(xm, ym, color="#666666", linewidth=0.55, alpha=0.55, zorder=2)
            if self._compute_mask is not None and self._compute_mask.size == n and self._compute_mask.any():
                idx = np.where(self._compute_mask)[0]
                breaks = np.where(np.diff(idx) > 1)[0]
                starts = np.concatenate([[0], breaks + 1])
                ends = np.concatenate([breaks + 1, [idx.size]])
                for s, e in zip(starts, ends):
                    seg = idx[s:e]
                    if seg.size == 0:
                        continue
                    xs, ys = _decimate_xy(t_ms[seg], y[seg])
                    self._ax_ref.plot(xs, ys, color="#7fdbff", linewidth=0.9, zorder=3)

            if self._ref_m1 is not None and self._ref_m2 is not None:
                xm1, ym1 = _decimate_xy(t_ms, self._ref_m1[:n])
                xm2, ym2 = _decimate_xy(t_ms, self._ref_m2[:n])
                self._ax_ref.plot(xm1, ym1, color="#9aa0a6", linewidth=0.45, alpha=0.75, label="M1", zorder=4)
                self._ax_ref.plot(xm2, ym2, color="#c4a35a", linewidth=0.45, alpha=0.75, label="M2", zorder=4)

            tx_ms = self._time_ms_at(k)
            if tx_ms is not None:
                self._ref_vline = self._ax_ref.axvline(
                    tx_ms, color="orangered", linewidth=1.4, linestyle="--", zorder=6
                )
                ri = int(np.argmin(np.abs(t_ms - tx_ms)))
                self._ax_ref.scatter(
                    [tx_ms],
                    [float(y[ri])],
                    s=28,
                    color="orangered",
                    zorder=7,
                    label="current time",
                )
            self._ax_ref.set_xlabel("time (ms) — click/drag to scrub")
            self._ax_ref.set_ylabel("mV")
            title = "Reference (M1 / M2 / M1−M2)" if self._ref_m1 is not None else "Reference signal"
            if tx_ms is not None:
                title += f"  —  t = {tx_ms:.2f} ms"
            self._ax_ref.set_title(title)
            self._ax_ref.legend(
                handles=[
                    Patch(facecolor="#3d8c40", alpha=0.35, label="computed (stim ±100 ms, SR ±50 ms)"),
                    Patch(facecolor="#555555", alpha=0.35, label="ignored"),
                ],
                loc="upper right",
                fontsize=7,
            )
            self._ax_ref.grid(True, alpha=0.3)
        else:
            self._ax_ref.set_title("M1/M2 reference unavailable for this acquisition")

    def _apply_time_index(self, k: int) -> None:
        if not self._patch_ready or self._S is None or self._dSdt is None:
            return
        self._set_time_index(k)

    def _on_display_mode(self) -> None:
        self._sync_color_limit_vars_from_global()
        if self._build_busy or not self._patch_ready:
            return
        try:
            k = int(self._time_var.get())
        except Exception:
            k = 0
        mode = self._display_mode_key()
        if mode in ("dvds", "d_dvds_dt") and not self._in_compute_window(k):
            first = self._first_compute_index()
            if first is not None:
                k = first
        # Defer mesh update so the radio-button handler returns before Laplacian solve.
        self.after_idle(lambda kk=k: self._apply_time_index(kk))

    def _on_slider(self, val) -> None:
        if self._build_busy:
            return
        try:
            k = int(round(float(val)))
        except (TypeError, ValueError):
            return
        if self._patch_ready:
            self._set_time_index(k)
        else:
            self._time_var.set(k)
            t_ms = self._time_ms_at(k)
            t_txt = f"  t={t_ms:.2f} ms" if t_ms is not None else ""
            self._time_lbl.config(text=f"sample {k}{t_txt}  —  build patch for mesh")
            self._draw_reference(k)
            self._draw_electrode_timeseries(None, k)
            self._canvas.draw_idle()

    def _end_viewer_preview(self) -> None:
        if self._viewer is not None and self._patch_on_viewer:
            try:
                self._restore_mesh_pick()
                self._viewer.clear_patch_preview_vectors()
                self._viewer.set_patch_preview_anchors(None)
                self._viewer.end_patch_preview()
            except Exception:
                traceback.print_exc()
            self._patch_on_viewer = False

    def _on_destroy(self, _evt=None) -> None:
        try:
            if getattr(self.app, "_acquisition_patch_window", None) is self:
                self.app._acquisition_patch_window = None
        except Exception:
            pass

    def _on_close(self) -> None:
        if self._build_busy:
            return
        self._end_viewer_preview()
        try:
            self.app.i = self._snap_i
            self.app.j = self._snap_j
            self.app.triple_active = self._snap_triple
            self.app.VT_active = self._snap_vt
            try:
                if self._snap_triple:
                    self.app.button_trip.config(text="Turn off Triple Extra Protocol")
                else:
                    self.app.button_trip.config(text="Switch to Triple Extra Protocol")
                if self._snap_vt:
                    self.app.button_VT.config(text="Turn off VT Protocol")
                else:
                    self.app.button_VT.config(text="Switch to VT Protocol")
            except Exception:
                pass
        except Exception:
            traceback.print_exc()
        self.destroy()


def open_acquisition_patch_window(app: "App") -> None:
    win = getattr(app, "_acquisition_patch_window", None)
    if win is not None:
        try:
            if win.winfo_exists():
                win.lift()
                win.focus_set()
                return
        except tk.TclError:
            pass
    app._acquisition_patch_window = AcquisitionPatchWindow(app)

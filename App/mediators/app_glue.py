"""Mount table / plots / mesh into the app shell and wire app ↔ UI glue."""

from __future__ import annotations

import traceback
from types import SimpleNamespace

import numpy as np
import tkinter as tk

from ..table_pkg import TableWidget
from ..shell import ShellFrame
from ..viewer3d import CartoMeshPanel


class TableAppGlue:
    """Sync ``TableWidget`` callbacks with ``App`` navigation/state."""

    def __init__(self, app) -> None:
        self.app = app

    def table_select_ctx(self, ctx):
        if ctx["row"] is None:
            return
        self.app.select([ctx["row"]])

    def table_move_ctx(self, ctx):
        if ctx["key"] == "Up":
            self.app.p_decrease()
        elif ctx["key"] == "Down":
            self.app.p_increase()

    def table_commit_ctx(self, ctx):
        row = ctx.get("row")
        col = ctx.get("col")
        val = ctx.get("value")
        if row is None:
            return

        i, j = self.app.to_i_j[row]
        idx = self.app.to_index[i][j]
        if col == 1:
            if self.app.triple_active:
                self.app.delta[idx][1] = val
            df_carto = self.app.carto.cont[i][0]
            df_carto.iat[j, df_carto.columns.get_loc("label_color")] = val

        tree = self.app.table.tree

        def follow_table_final_row():
            cur_iid = tree.cur_iid or (tree.selection()[0] if tree.selection() else None)
            if not cur_iid:
                return
            new_row = tree._row_index_from_iid(cur_iid)
            if new_row is None:
                return
            self.app.i, self.app.j = self.app.to_i_j[new_row]
            self.app.update_plot()

        tree.after_idle(follow_table_final_row)


class AppLayoutGlue:
    """Fill ``ShellFrame.panel_hosts`` with table, signal plots, and 3D mesh."""

    def __init__(self, app) -> None:
        self.app = app
        self.frame: ShellFrame | None = None
        self.dock = None
        self.sections: dict[str, tk.Frame] = {}
        self.grid_main: tk.Frame | None = None
        self.table = None
        self.frame1: tk.Frame | None = None
        self.frame_mesh: tk.Frame | None = None
        self.mesh_panel = None
        self._mesh_placeholder: tk.Label | None = None
        self._mesh_init_done = False
        self._plots_layout_size: tuple[int, int] | None = None

    def mount(self, frame: ShellFrame) -> None:
        """Attach feature widgets to the empty panel hosts from :func:`build_shell_frame`."""
        self.frame = frame
        self.grid_main = frame["grid_main"]
        self.dock = frame["dock_grid"]
        self.sections = frame["sections"]
        hosts = frame["panel_hosts"]

        self._build_table(hosts["table"])
        self._build_plots(hosts["plots"])
        self._build_mesh(hosts["mesh"])

        self.dock.set_layout_change_callback(self._on_panel_layout_changed)

        self.app.grid_main = self.grid_main
        self.app.dock_grid = self.dock
        self.app.sections = self.sections
        self.app.table = self.table
        self.app.frame1 = self.frame1
        self.app.frame_mesh = self.frame_mesh
        self.app.mesh_panel = self.mesh_panel

    def _on_panel_layout_changed(self, panel_id: str) -> None:
        if panel_id == "plots":
            self._plots_layout_size = None
            self._relayout_plots()
        elif panel_id == "mesh":
            panel = self.mesh_panel
            viewer = getattr(panel, "viewer", None) if panel is not None else None
            if viewer is not None:
                viewer.redraw()
        elif panel_id == "table":
            self._relayout_table()

    def _relayout_table(self) -> None:
        if self.table is None:
            return
        self.table.update_idletasks()

    def _relayout_plots(self) -> None:
        canvas = getattr(self.app, "canvas", None)
        if canvas is None:
            return
        tk_widget = canvas.get_tk_widget()
        tk_widget.update_idletasks()
        w = max(int(tk_widget.winfo_width()), 1)
        h = max(int(tk_widget.winfo_height()), 1)
        size = (w, h)
        if self._plots_layout_size == size:
            return
        if getattr(self.app, "_resize_paused", False):
            self.app._pending_canvas_draw = True
            return
        self._plots_layout_size = size
        canvas.resize(SimpleNamespace(width=w, height=h))
        presenter = getattr(self.app, "plot_presenter", None)
        if presenter is not None:
            try:
                presenter.restore_axes_view()
            except Exception:
                traceback.print_exc()
        try:
            canvas.draw_idle()
        except Exception:
            traceback.print_exc()

    def _build_table(self, parent: tk.Frame) -> None:
        self.table = TableWidget(parent, self.app.Table)
        self.table.pack(fill="both", expand=True)
        self.table.tree.defaults = {1: ["Reject", "POS", "NEG"]}
        try:
            self.app._sync_original_label_column()
        except Exception:
            traceback.print_exc()

    def _build_plots(self, parent: tk.Frame) -> None:
        self.frame1 = tk.Frame(parent, pady=5, background="white")
        self.frame1.pack(fill="both", expand=True)

    def _build_mesh(self, parent: tk.Frame) -> None:
        self.frame_mesh = tk.Frame(parent, background="black")
        self.frame_mesh.pack(fill="both", expand=True)
        self._mesh_placeholder = tk.Label(
            self.frame_mesh,
            text="Loading 3D viewer…",
            bg="black",
            fg="#666666",
        )
        self._mesh_placeholder.pack(expand=True)

    def finish_deferred_mesh(self) -> None:
        """Create the heavy OpenGL panel after the main window is shown."""
        if self._mesh_init_done or self.frame_mesh is None:
            return
        self._mesh_init_done = True
        if self._mesh_placeholder is not None:
            try:
                self._mesh_placeholder.destroy()
            except tk.TclError:
                pass
            self._mesh_placeholder = None
        try:
            self.mesh_panel = CartoMeshPanel(self.frame_mesh, self.app.carto)
            self.mesh_panel.pack(fill="both", expand=True)
            self.app.mesh_panel = self.mesh_panel
            try:
                self.app.mesh_glue.attach(self.mesh_panel)
            except Exception:
                traceback.print_exc()
            ready = getattr(self.app, "on_mesh_panel_ready", None)
            if callable(ready):
                ready()
        except Exception as exc:
            traceback.print_exc()
            tk.Label(
                self.frame_mesh,
                text=f"3D viewer disabled: {exc}",
                bg="black",
                fg="white",
                wraplength=240,
                justify="left",
            ).pack(fill="both", expand=True, padx=8, pady=8)


class MeshAppGlue:
    def __init__(self, app) -> None:
        self.app = app
        self.panel = None
        self.viewer = None
        self._attached = False

    def attach(self, panel) -> None:
        self.panel = panel
        if panel is None:
            return
        self.viewer = panel.viewer
        if self.viewer is None:
            return
        self.viewer.on_pick_callback = self._on_pick_callback
        try:
            self.viewer.set_delta_provider(self)
        except Exception:
            traceback.print_exc()
        self._populate_electrodes()
        self.sync_from_app()
        self._attached = True

    def _populate_electrodes(self) -> None:
        if self.viewer is None:
            return
        positions: list[tuple[float, float, float]] = []
        gindices: list[int] = []
        labels: list[str] = []
        for global_idx, ij in enumerate(self.app.to_i_j):
            i, j = ij
            try:
                df = self.app.carto.cont[i][0]
                row = df.iloc[int(j)]
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
            except Exception:
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            try:
                pn = row["point number"]
            except Exception:
                pn = ""
            positions.append((x, y, z))
            gindices.append(int(global_idx))
            labels.append(f"P{pn}")

        if not positions:
            return
        try:
            self.viewer.set_electrodes(
                np.asarray(positions, dtype=np.float64),
                gindices,
                labels,
            )
        except Exception:
            traceback.print_exc()

    def sync_from_app(self) -> None:
        if self.viewer is None:
            return
        try:
            i, j = int(self.app.i), int(self.app.j)
            global_idx = self.app.to_index[i][j]
            self.viewer.set_selected_global_index(int(global_idx))
        except Exception:
            traceback.print_exc()

    def _on_pick_callback(self, kind: str, payload, info: dict) -> None:
        try:
            self.on_pick(kind, payload, info)
        except Exception:
            traceback.print_exc()

    def on_pick(self, kind: str, payload, info: dict) -> None:
        if kind != "sphere" or payload is None:
            return
        global_idx = int(payload)
        try:
            self.app.select([global_idx])
        except Exception:
            traceback.print_exc()

    _SKIP_LIST_KEYS = frozenset({"stim", "sinus"})
    _MAX_PER_KIND_SUFFIX = (
        ("_stim", 3),
        ("_sinus", 1),
    )

    @classmethod
    def _index_cap_for(cls, key: str) -> int | None:
        for suffix, cap in cls._MAX_PER_KIND_SUFFIX:
            if key.endswith(suffix):
                return cap
        return None

    @staticmethod
    def _entry_metrics(entry) -> dict | None:
        if entry is None or entry == 0:
            return None
        if hasattr(entry, "metrics") and isinstance(getattr(entry, "metrics"), dict):
            return entry.metrics
        try:
            metrics = entry[2]
        except Exception:
            return None
        return metrics if isinstance(metrics, dict) else None

    @staticmethod
    def _to_finite_float(x) -> float:
        if x is None:
            return float("nan")
        if isinstance(x, bool):
            return float("nan")
        if isinstance(x, (list, tuple)):
            return float("nan")
        if isinstance(x, np.ndarray):
            if x.size != 1:
                return float("nan")
            try:
                v = float(x.item())
            except (TypeError, ValueError):
                return float("nan")
            return v if np.isfinite(v) else float("nan")
        try:
            v = float(x)
        except (TypeError, ValueError):
            return float("nan")
        return v if np.isfinite(v) else float("nan")

    _LAT_DERIVATIONS = (
        ("stim", "refs_stim", "lat_stim"),
        ("sinus", "refs_sinus", "lat_sinus"),
    )

    def _iter_entry_scalars(self, entry):
        metrics = self._entry_metrics(entry)
        if metrics is None:
            return
        for key, val in metrics.items():
            ks = str(key)
            if ks in self._SKIP_LIST_KEYS:
                continue
            if isinstance(val, (list, tuple)) or (
                isinstance(val, np.ndarray) and val.ndim >= 1 and val.size != 1
            ):
                cap = self._index_cap_for(ks)
                for i, sub in enumerate(val):
                    if cap is not None and i >= cap:
                        break
                    v = self._to_finite_float(sub)
                    if np.isfinite(v):
                        yield f"{ks}[{i + 1}]", v
            else:
                v = self._to_finite_float(val)
                if np.isfinite(v):
                    yield ks, v

        for win_key, ref_key, out_key in self._LAT_DERIVATIONS:
            windows = metrics.get(win_key)
            refs = metrics.get(ref_key)
            if not isinstance(windows, (list, tuple, np.ndarray)):
                continue
            if not isinstance(refs, (list, tuple, np.ndarray)):
                continue
            n = min(len(windows), len(refs))
            cap = self._index_cap_for(out_key)
            if cap is not None:
                n = min(n, cap)
            for i in range(n):
                w = windows[i]
                if not isinstance(w, (list, tuple, np.ndarray)) or len(w) < 1:
                    continue
                try:
                    start = float(w[0])
                    ref = float(refs[i])
                except (TypeError, ValueError):
                    continue
                lat = start - ref
                if np.isfinite(lat):
                    yield f"{out_key}[{i + 1}]", lat

    def get_delta_metric_keys(self) -> list[str]:
        keys: set[str] = set()
        for entry in getattr(self.app, "delta", []) or []:
            for tag, _v in self._iter_entry_scalars(entry):
                keys.add(tag)
        return sorted(keys, key=self._sort_key)

    @staticmethod
    def _sort_key(tag: str):
        bracket = tag.find("[")
        if bracket < 0:
            return (tag, -1)
        try:
            idx = int(tag[bracket + 1 : -1])
        except ValueError:
            idx = -1
        return (tag[:bracket], idx)

    def get_delta_values_for(self, key: str) -> dict[int, float]:
        out: dict[int, float] = {}
        delta = getattr(self.app, "delta", None) or []
        to_ij = getattr(self.app, "to_i_j", None) or []
        for gidx, entry in enumerate(delta):
            try:
                i, j = to_ij[gidx]
                lab = str(self.app.carto.cont[i][0].loc[int(j), "label_color"]).strip().lower()
            except Exception:
                lab = ""
            if lab == "reject":
                continue
            for tag, v in self._iter_entry_scalars(entry):
                if tag == key:
                    out[gidx] = v
                    break
        return out

    def get_electrode_label_color(self, global_idx: int) -> str:
        """Return ``label_color`` (POS / NEG / Reject) for a flat table row index."""
        try:
            i, j = self.app.to_i_j[int(global_idx)]
            return str(self.app.carto.cont[i][0].loc[int(j), "label_color"]).strip()
        except Exception:
            return ""

    def notify_delta_changed(self, global_idx=None) -> None:
        if self.viewer is None:
            return
        try:
            self.viewer.notify_delta_changed(global_idx)
        except Exception:
            traceback.print_exc()

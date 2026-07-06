"""Main app organizer: glue subpackages via mediators + app logic (ML, delta, navigation)."""

# Import order pattern:
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import ImageGrab
import pandas as pd
from tkinter import filedialog
from tkinter import messagebox
import threading
import traceback


# private libs
from .carto import Carto
from .core import AppController, DeltaStore, SessionStore, register_app
from .mediators.wiring import DEFAULT_WIRING
from .mediators.tools_ribbon import build_left_ribbon
from .ml import (
    MLBundle,
    find_local_model_files,
    load_ml_bundle,
    predicted_label_for_delta,
    should_reject_delta_entry,
)
from .mediators.resize_pause import (
    attach_resize_pause,
    install_canvas_draw_guard,
    register_resize_watch_widgets,
)
from .plotting.figure_host import SignalFigureHost
from .plotting.spectrogram_settings import SpectrogramSettings
from .plotting.stft_settings_dialog import StftSettingsDialog
from .shell import build_shell_frame


def _init_startup_state(app, name="Saman"):
    return {
        "name": name,
        "cont": app.carto.cont,
        "forcefull": False,
        "is_running": True,
        "triple_active": False,
        "VT_active": False,
        "start_x_y": [],
        "direction": 1,
        "ml_bundle": None,
        "ml_selected_models": [],
        "show_original_labels": False,
    }


class App(tk.Tk):
    def __init__(self, carto: Carto = None, *, wiring=DEFAULT_WIRING):
        register_app(self)
        self.wiring = wiring
        self.plot_settings = SpectrogramSettings()
        self.carto = carto
        self.session_store = SessionStore()
        self.delta_store = DeltaStore(carto, session_store=self.session_store)
        self.delta_store.build_index_map(hide_labels=True)
        self.delta_store.allocate_entries()
        self.Table = self.delta_store.build_table_dataframe()
        self.controller = AppController(self)
        self.plot_presenter = wiring.presenter_cls(self)
        # Mesh re-interp must run AFTER the plot has actually finished writing
        # to ``self.delta`` (otherwise it would interpolate against stale
        # values). The async triple-extra path makes this listener mandatory.
        try:
            self.plot_presenter.add_plot_done_listener(self._on_plot_done)
        except Exception:
            traceback.print_exc()
        self.table_glue = wiring.table_glue_cls(self)
        self.mesh_glue = wiring.mesh_glue_cls(self)
        self.i, self.j = 0, 0

    @property
    def delta(self):
        return self.delta_store.delta

    @delta.setter
    def delta(self, value):
        self.delta_store.delta = value

    @property
    def to_index(self):
        return self.delta_store.to_index

    @property
    def to_i_j(self):
        return self.delta_store.to_i_j

    @property
    def labels_memory(self):
        return self.delta_store.labels_memory

    def start(self, name="Saman"):

        # super() calls parent class behavior (tk.Tk initialization) before custom setup.
        super().__init__()
        print("start")
        startup = _init_startup_state(self, name)
        self.name = startup["name"]
        self.cont = startup["cont"]
        self.forcefull = startup["forcefull"]
        self.is_running = startup["is_running"]
        self.triple_active = startup["triple_active"]
        self.VT_active = startup["VT_active"]
        self.start_x_y = startup["start_x_y"]
        self.direction = startup["direction"]
        self.ml_bundle = startup["ml_bundle"]
        self.ml_selected_models = startup["ml_selected_models"]
        self.show_original_labels = startup["show_original_labels"]
        self._compute_all_running = False
        self._compute_all_queue = []
        self._compute_all_saved_ij = (0, 0)
        # ---- main window size and title
        self.geometry("900x700")
        self.title(f"My {self.name} APP")

        # ---- main window frame (ribbon + dock) then mount panels via glue
        frame = build_shell_frame(self)
        self.layout_glue = self.wiring.layout_glue_cls(self)
        self.layout_glue.mount(frame)

        ribbon = build_left_ribbon(self, frame["ribbon_column"], frame["dock_grid"])
        self.label = ribbon["label"]
        self.check_boxes = ribbon["check_boxes"]
        self.button_trip = ribbon["button_trip"]
        self.button_VT = ribbon["button_VT"]
        self.button_dropdown = ribbon["button_dropdown"]
        self.button_compute_all = ribbon["button_compute_all"]
        self.button_compute_cv = ribbon["button_compute_cv"]
        self.ml_model_var = ribbon["ml_model_var"]
        self.ml_combo = ribbon["ml_combo"]
        self.pred_label = ribbon["pred_label"]
        self.show_original_labels_var = ribbon["show_original_labels_var"]
        self.panel_vis_vars = ribbon.get("panel_vis_vars", {})
        # ---- scan model/ folder for joblib bundles (after table exists)
        # Auto-discover joblib bundles in the sibling ``model/`` folder so the
        # dropdown can offer them without forcing the user to file-dialog
        # every time. Selection stays at "(none)" until the user picks one.
        self._refresh_ml_discovery()

        # ---- create matplotlib figure and axes
        # self.axes is a dictionary that contains the axes objects for the top, mid, and bottom axes.
        self.set_figure()
        install_canvas_draw_guard(self)
        # Only pause redraws for outer window / ribbon sash drags — not dock panel moves.
        resize_watch = [
            self,
            frame["ribbon_shell"],
            frame["content_host"],
        ]
        self._shell_outer_pane = frame["outer_pane"]
        attach_resize_pause(
            self,
            panes=[self._shell_outer_pane],
            watch_widgets=resize_watch,
        )
        # ---- legend canvases under plots
        self._build_legend_canvases()
        # ---- keyboard / mouse bindings
        self.main()
        # ---- initial plot draw
        self.plot()
        if getattr(self.layout_glue, "_relayout_plots", None) is not None:
            self.layout_glue._plots_layout_size = None
            self.layout_glue._relayout_plots()
        # Heavy OpenGL mesh init runs after the window is shown.
        self.after_idle(self._deferred_mesh_init)


    def _deferred_mesh_init(self) -> None:
        glue = getattr(self, "layout_glue", None)
        if glue is not None:
            glue.finish_deferred_mesh()

    def on_mesh_panel_ready(self) -> None:
        panel = getattr(self, "mesh_panel", None)
        if panel is None:
            return
        viewer = getattr(panel, "viewer", None)
        mesh_ribbon = getattr(panel, "ribbon_pane", None)
        register_resize_watch_widgets(
            self,
            [w for w in (viewer, mesh_ribbon) if w is not None],
        )

    def capture_window(self,name=None):
            root=self
            if name is None:
                name=f"screenshot{root.cont[root.i][0].reset_index(drop=True)['point number'][root.j]}.png"
            x = root.winfo_rootx()
            y = root.winfo_rooty()
            width = root.winfo_width()
            height = root.winfo_height()    #get details about window
            takescreenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
            takescreenshot.save(name,dpi=(1920,1080))

    def triple_protocol(self,event=None):

        self.triple_active=not(self.triple_active)
        if self.triple_active:
            self.button_trip.config(text="Turn off Triple Extra Protocol")
        else:
            self.button_trip.config(text="Switch to Triple Extra Protocol")
        self.update_plot()
    def VT_protocol(self,event=None):

        self.VT_active=not(self.VT_active)
        if self.VT_active:
            self.button_VT.config(text="Turn off VT Protocol")
        else:
            self.button_VT.config(text="Switch to VT Protocol")
        self.update_plot()
        
    def set_figure(self):
        """(Re)build the matplotlib figure embedded in ``frame1``."""
        def _on_canvas(canvas):
            install_canvas_draw_guard(self)
            glue = getattr(self, "layout_glue", None)
            if glue is not None:
                glue._plots_layout_size = None

        host = SignalFigureHost(self.frame1, on_canvas_created=_on_canvas)
        host.attach_to(self)
        self._figure_host = host

    def _build_legend_canvases(self):
        host = getattr(self, "_figure_host", None)
        if host is not None:
            host.build_legend_canvases()
            host.attach_to(self)

        

    def _bind_canvas_events(self):
        """Connect matplotlib canvas events. Re-run whenever the canvas is
        rebuilt (e.g. after ``set_figure``)."""
        def mpl_arrow(event):
            if event.key=="right":
                self.s_increase(event)
            elif event.key=="left":
                self.s_decrease(event)
            elif event.key=="up":
                self.p_decrease(event)
            elif event.key=="down":
                self.p_increase(event)

        self.canvas.mpl_connect("key_press_event",mpl_arrow)
        self.canvas.mpl_connect("button_press_event", self.on_right_click)
        self.canvas.mpl_connect("button_release_event", self.on_right_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

    def main(self):
        self._bind_canvas_events()
        tv = self.table.tree

        # Event callback hooks:
        # the table calls these lambdas when user interactions occur.
        # This is event-driven programming (UI reacts to events).
        # row selection (mouse click / Up/Down changes selection)
        tv.on_select_row = lambda ctx: self.table_glue.table_select_ctx(ctx)

        # cell move (Up/Down/Left/Right/Enter triggers this hook)
        tv.on_cell_move = lambda ctx: self.table_glue.table_move_ctx(ctx)

        # edit committed (Return, dropdown pick, focus-out if you set it to commit)
        tv.on_edit_commit = lambda ctx: self.table_glue.table_commit_ctx(ctx)

        super().protocol("WM_DELETE_WINDOW", self.quit)
        # Create the figure and axis objects
    def quit(self):
        self.is_running = False
        super().quit()
        super().destroy() 


    def checker(self):
        self.update_plot()

    def _original_labels_flat(self) -> list:
        return self.delta_store.original_labels_flat()

    def _sync_original_label_column(self) -> None:
        """Fill ``original_label`` and show/hide the pre-allocated column."""
        col = "original_label"
        table = getattr(self, "table", None)
        if table is None:
            return
        tv = table.tree
        labels = self._original_labels_flat()
        if col not in list(tv["columns"]):
            return
        tv.update_column_values(col, labels)
        if col in self.Table.columns:
            self.Table[col] = labels
        tv.set_column_visible(col, bool(self.show_original_labels))

    def _toggle_original_labels(self) -> None:
        self.show_original_labels = bool(self.show_original_labels_var.get())
        self._sync_original_label_column()


    def on_right_click(self, event):
        # Delegation: App forwards plotting interactions to PlotPresenter.
        # Delegation separates UI shell responsibilities from plotting logic.
        return self.plot_presenter.on_right_click(event)


    def on_right_follow_mouse(self,event):
        return self.plot_presenter.on_right_follow_mouse(event)
    

    def on_right_release(self, event):
        return self.plot_presenter.on_right_release(event)
 
    def on_scroll(self, event):
        return self.plot_presenter.on_scroll(event)

    def plot_main(self,ax,x,y2,arg="top",reff=[]):
        return self.plot_presenter.plot_main(ax=ax, x=x, y2=y2, arg=arg, reff=reff)
        

    def plot(self):
        return self.plot_presenter.plot()
        
            
     
    
    def select(self,event):
        i,j=self.to_i_j[event[0]]
        print("selected row with select binding function",event[0])
        self.i=i
        self.j=j
        self.update_plot()

    
       
    def _on_plot_done(self) -> None:
        """Called after PlotPresenter finishes drawing (sync or async)."""
        try:
            if getattr(self, "mesh_glue", None) is not None:
                self.mesh_glue.notify_delta_changed()
        except Exception:
            traceback.print_exc()

    def _compute_all_clicked(self) -> None:
        if self._compute_all_running:
            return
        n = len(self.to_i_j)
        if n <= 0:
            return
        self._compute_all_saved_ij = (int(self.i), int(self.j))
        self._compute_all_queue = list(range(n))
        self._compute_all_running = True
        try:
            self.button_compute_all.config(state=tk.DISABLED)
        except Exception:
            traceback.print_exc()
        try:
            self.config(cursor="watch")
            self.update_idletasks()
        except Exception:
            traceback.print_exc()
        self._compute_all_run_next()

    def _compute_all_run_next(self) -> None:
        if not self._compute_all_running:
            return
        if not self._compute_all_queue:
            self._compute_all_finish()
            return
        row_idx = int(self._compute_all_queue.pop(0))
        try:
            self.plot_presenter.start_compute_all_row(
                row_idx, after_apply=self._compute_all_run_next
            )
        except Exception:
            traceback.print_exc()
            self.after(0, self._compute_all_run_next)

    def _compute_all_finish(self) -> None:
        self._compute_all_running = False
        try:
            self.button_compute_all.config(state=tk.NORMAL)
        except Exception:
            traceback.print_exc()
        try:
            self.config(cursor="")
        except Exception:
            traceback.print_exc()
        si, sj = self._compute_all_saved_ij
        self.i, self.j = int(si), int(sj)
        try:
            self.update_plot(forcefull=False)
        except Exception:
            traceback.print_exc()

    def update_plot(self,forcefull=False):
        """Update UI for the current selection.

        Heavy compute (filters, librosa spectrograms, deflection peaks, the
        whole triple-extra pipeline) is delegated to a background worker
        inside ``PlotPresenter``; this method only does the cheap UI sync
        that the user expects to feel instant. The mesh delta re-interp is
        triggered once the heavy plot apply step completes, via the plot-
        done listener registered in ``main_app.start()``.
        """
        if self._compute_all_running:
            self.forcefull = bool(forcefull)
            try:
                point_number = self.cont[self.i][0].reset_index(drop=True)["point number"][self.j]
                self.label.config(text=f"point {point_number}")
            except Exception:
                traceback.print_exc()
            index = self.to_index[self.i][self.j]
            try:
                self._update_pred_label_for_index(index)
            except Exception:
                traceback.print_exc()
            try:
                self.table.tree.see(f"row{index}")
                self.table.tree.selection_set(f"row{index}")
                self.table.tree.focus(f"row{index}")
                self.table.tree.cur_iid = f"row{index}"
            except Exception:
                traceback.print_exc()
            try:
                if getattr(self, "mesh_glue", None) is not None:
                    self.mesh_glue.sync_from_app()
            except Exception:
                traceback.print_exc()
            return

        self.forcefull = bool(forcefull)
        presenter = getattr(self, "plot_presenter", None)
        if presenter is not None:
            try:
                presenter.capture_axes_view()
            except Exception:
                traceback.print_exc()
        # Cheap UI sync first so the click feels instant even on big sessions.
        for i in self.axes.values():
            i.clear()
        if presenter is not None:
            try:
                presenter.restore_axes_view()
            except Exception:
                traceback.print_exc()
        try:
            point_number = self.cont[self.i][0].reset_index(drop=True)["point number"][self.j]
            self.label.config(text=f"point {point_number}")
        except Exception:
            traceback.print_exc()
        index = self.to_index[self.i][self.j]
        # Show the cached prediction (if any) immediately on navigation; the
        # persist hook will refresh it once the heavy plot finishes.
        try:
            self._update_pred_label_for_index(index)
        except Exception:
            traceback.print_exc()
        print("current index", index)
        try:
            self.table.tree.see(f"row{index}")
            self.table.tree.selection_set(f"row{index}")
            self.table.tree.focus(f"row{index}")
            self.table.tree.cur_iid = f"row{index}"
        except Exception:
            traceback.print_exc()
        # Update the 3D viewer's highlighted sphere immediately (cheap).
        try:
            if getattr(self, "mesh_glue", None) is not None:
                self.mesh_glue.sync_from_app()
        except Exception:
            traceback.print_exc()
        # Schedule the heavy plot. Mesh delta re-interp is fired by the
        # plot-done listener once the triple-extra worker finishes.
        try:
            self.plot()
        except Exception:
            traceback.print_exc()
        if presenter is not None:
            try:
                presenter.restore_axes_view()
            except Exception:
                traceback.print_exc()
    def p_increase(self,event=None):
        return self.controller.p_increase(event)

    def p_decrease(self,event=None):
        return self.controller.p_decrease(event)

    def s_increase(self,event=None):
        return self.controller.s_increase(event)


    def s_decrease(self,event=None):
        return self.controller.s_decrease(event)

    def _save_delta_dialog(self):
        file_path = filedialog.asksaveasfilename(
            confirmoverwrite=True,
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if file_path:
            self.delta_store.save_to(file_path)
            print(f"File is saved in {file_path}")

    def _load_delta_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not file_path:
            return
        print(f"Data loaded from {file_path}")
        self.delta_store.load_from(file_path)
        self.delta_store.refresh_table_columns(self)
        try:
            if getattr(self, "mesh_glue", None) is not None:
                self.mesh_glue.notify_delta_changed()
        except Exception:
            traceback.print_exc()

    def _open_meta_parameters_window(self):
        top_window = tk.Toplevel()
        top_window.attributes("-topmost", True)
        StftSettingsDialog(top_window, self)

    def drop_down(self):
        menu=tk.Menu(master=self.frame1,tearoff=False)
        menu.add_command(label="Save_Delta",command=self._save_delta_dialog)
        menu.add_command(label="Load_Delta",command=self._load_delta_dialog)
        menu.add_command(label="meta_parameters",command=self._open_meta_parameters_window)
        menu.tk_popup(self.button_dropdown.winfo_rootx(),self.button_dropdown.winfo_rooty()+self.button_dropdown.winfo_height())
    def create_legend(self,leg,canvas,addition=None):
        return self.plot_presenter.create_legend(leg=leg, canvas=canvas, addition=addition)
    def Energy(self,ax,x,y,legends=None):   
        return self.plot_presenter.energy(ax=ax, x=x, y=y, legends=legends)

    # ------------------------------------------------------------------ ML
    # Bundle loading, dropdown wiring, and per-row prediction. The bundle
    # itself is built by ``model/ml_model_io.save_sklearn_model_suite`` —
    # a joblib pickle containing every sklearn pipeline plus metadata.

    def _ml_path_candidates(self) -> list:
        try:
            return [str(path) for path in find_local_model_files()]
        except Exception:
            traceback.print_exc()
            return []

    def _refresh_ml_discovery(self) -> None:
        """Repopulate the dropdown if no bundle has been loaded yet."""
        if self.ml_bundle is not None:
            return
        paths = self._ml_path_candidates()
        if not paths:
            return
        try:
            self.ml_bundle = load_ml_bundle(paths[0])
        except Exception:
            traceback.print_exc()
            self.ml_bundle = None
            return
        self._populate_ml_combo(self.ml_bundle.model_names, select_first=False)

    def _populate_ml_combo(self, names, *, select_first: bool) -> None:
        try:
            values = ["(none)"] + list(names)
            self.ml_combo["values"] = values
        except Exception:
            traceback.print_exc()
            return
        if select_first and names:
            self.ml_model_var.set(names[0])
        else:
            self.ml_model_var.set("(none)")
        self._on_ml_model_change()

    def _open_ml_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Load ML model bundle",
            filetypes=[("Joblib bundle", "*.joblib"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            self.ml_bundle = load_ml_bundle(path)
        except Exception:
            traceback.print_exc()
            return
        # Auto-select the first model on explicit load so the user sees
        # predictions immediately. Discovery on startup stays at "(none)"
        # so we never silently change behaviour.
        self._populate_ml_combo(self.ml_bundle.model_names, select_first=True)

    def _on_ml_model_change(self, *_event) -> None:
        val = self.ml_model_var.get()
        if not val or val == "(none)" or self.ml_bundle is None:
            self.ml_selected_models = []
            self._clear_prediction_column()
            self._clear_pos_proba_in_deltas()
            if getattr(self, "pred_label", None) is not None:
                self.pred_label.config(text="Predicted: —")
            return
        self.ml_selected_models = [val]
        self._refresh_all_predictions()
        try:
            cur = self.to_index[self.i][self.j]
            self._update_pred_label_for_index(cur)
        except Exception:
            traceback.print_exc()

    def _clear_pos_proba_in_deltas(self) -> None:
        if self.delta_store.clear_all_pos_proba():
            try:
                if getattr(self, "mesh_glue", None) is not None:
                    self.mesh_glue.notify_delta_changed()
            except Exception:
                traceback.print_exc()

    def _clear_prediction_column(self) -> None:
        table = getattr(self, "table", None)
        if table is None:
            return
        try:
            col_loc = self.Table.columns.get_loc("Prediction")
        except Exception:
            return
        self.Table.iloc[:, col_loc] = ""
        try:
            tv = table.tree
            cols = list(tv["columns"])
            if "Prediction" not in cols:
                return
            pred_col = cols.index("Prediction")
            for idx in range(len(self.delta)):
                iid = f"row{idx}"
                vals = list(tv.item(iid).get("values") or [])
                if 0 <= pred_col < len(vals):
                    vals[pred_col] = ""
                    tv.item(iid, values=vals)
        except Exception:
            traceback.print_exc()

    def _update_pred_label_for_index(self, idx: int) -> None:
        if getattr(self, "pred_label", None) is None:
            return
        if not self.ml_selected_models:
            self.pred_label.config(text="Predicted: —")
            return
        try:
            col_loc = self.Table.columns.get_loc("Prediction")
            value = str(self.Table.iat[idx, col_loc] or "")
        except Exception:
            traceback.print_exc()
            value = ""
        if value:
            self.pred_label.config(text=f"Predicted: {value}")
        else:
            self.pred_label.config(text="Predicted: (not computed)")

    def _mark_row_reject(self, idx: int, c1) -> None:
        """Force ``Reject`` on carto label, delta, table cell. Idempotent."""
        try:
            self.delta_store.set_carto_label(idx, "Reject")
        except Exception:
            traceback.print_exc()
        try:
            self.delta_store.set_entry_label(idx, "Reject")
        except Exception:
            traceback.print_exc()
        if isinstance(c1, dict):
            c1.pop("pos_proba", None)
        try:
            tv = self.table.tree
            cols = list(tv["columns"])
            if "label_color" in cols:
                lc_col = cols.index("label_color")
                iid = f"row{idx}"
                vals = list(tv.item(iid).get("values") or [])
                if 0 <= lc_col < len(vals):
                    vals[lc_col] = "Reject"
                    tv.item(iid, values=vals)
        except Exception:
            traceback.print_exc()

    def _predict_proba_for_entry(self, entry) -> float | None:
        """Positive-class probability for one delta entry, or None."""
        from .ml import extract_features_from_delta_entry

        feats = extract_features_from_delta_entry(entry)
        if feats is None or self.ml_bundle is None or not self.ml_selected_models:
            return None
        try:
            result = self.ml_bundle.predict(
                feats, self.ml_selected_models, strategy="average_proba"
            )
        except ValueError:
            # Selected model(s) have no predict_proba — fall back to the
            # hard vote so the mesh map at least shows 0 / 1 anchors.
            try:
                result = self.ml_bundle.predict(
                    feats, self.ml_selected_models, strategy="vote"
                )
            except Exception:
                traceback.print_exc()
                return None
            pred = result.get("predictions")
            if pred is None or len(pred) == 0:
                return None
            return 1.0 if bool(pred[0]) else 0.0
        except Exception:
            traceback.print_exc()
            return None
        proba = result.get("positive_probabilities")
        if proba is None or len(proba) == 0:
            return None
        try:
            return float(proba[0])
        except (TypeError, ValueError):
            return None

    def _write_prediction_cell(self, idx: int, label: str) -> None:
        try:
            col_loc = self.Table.columns.get_loc("Prediction")
            self.Table.iat[idx, col_loc] = label
        except Exception:
            traceback.print_exc()
        try:
            tv = self.table.tree
            cols = list(tv["columns"])
            if "Prediction" in cols:
                pred_col = cols.index("Prediction")
                iid = f"row{idx}"
                vals = list(tv.item(iid).get("values") or [])
                if 0 <= pred_col < len(vals):
                    vals[pred_col] = label
                    tv.item(iid, values=vals)
        except Exception:
            traceback.print_exc()
        try:
            cur = self.to_index[self.i][self.j]
        except Exception:
            cur = -1
        if cur == idx:
            self._update_pred_label_for_index(idx)

    def _predict_row(self, idx: int, *, notify_mesh: bool = True) -> None:
        """Compute, store, and display the prediction for one table row."""
        if self.ml_bundle is None or not self.ml_selected_models:
            return
        try:
            entry = self.delta[idx]
        except Exception:
            return
        if entry == 0:
            self._write_prediction_cell(idx, "")
            return
        c1 = entry[2] if isinstance(entry, list) and len(entry) >= 3 else None
        if should_reject_delta_entry(entry):
            self._mark_row_reject(idx, c1)
            self._write_prediction_cell(idx, "Reject")
            if notify_mesh:
                try:
                    if getattr(self, "mesh_glue", None) is not None:
                        self.mesh_glue.notify_delta_changed(idx)
                except Exception:
                    traceback.print_exc()
            return
        label = predicted_label_for_delta(
            self.ml_bundle, self.ml_selected_models, entry
        )
        proba = self._predict_proba_for_entry(entry)
        if isinstance(c1, dict):
            if proba is None:
                c1.pop("pos_proba", None)
            else:
                c1["pos_proba"] = float(min(1.0, max(0.0, proba)))
        self._write_prediction_cell(idx, label)
        if notify_mesh:
            try:
                if getattr(self, "mesh_glue", None) is not None:
                    self.mesh_glue.notify_delta_changed(idx)
            except Exception:
                traceback.print_exc()

    def _refresh_all_predictions(self) -> None:
        """Re-label every row, then notify the mesh exactly once.

        We suppress the per-row mesh notification (which would otherwise
        kick off a full ``_compute_delta_interpolated`` for every single
        electrode) and fire a single global notify at the end so the
        ``delta:pos_proba`` field appears in the 3D dropdown and the
        interpolated POS/NEG map is computed once.
        """
        if self.ml_bundle is None or not self.ml_selected_models:
            return
        for idx in range(len(self.delta)):
            self._predict_row(idx, notify_mesh=False)
        try:
            if getattr(self, "mesh_glue", None) is not None:
                self.mesh_glue.notify_delta_changed()
        except Exception:
            traceback.print_exc()

    def _compute_conduction_velocity_clicked(self) -> None:
        """Build CV magnitude maps for S1/S2/S3 stims and SR LAT."""
        if self.mesh_panel is None or getattr(self.mesh_panel, "viewer", None) is None:
            return
        viewer = self.mesh_panel.viewer
        try:
            self.button_compute_cv.config(state=tk.DISABLED)
            self.config(cursor="watch")
            self.update_idletasks()
        except Exception:
            traceback.print_exc()
        produced: dict[str, int] = {}
        try:
            produced = viewer.compute_conduction_velocity() or {}
        except Exception:
            traceback.print_exc()
        finally:
            try:
                self.button_compute_cv.config(state=tk.NORMAL)
                self.config(cursor="")
            except Exception:
                traceback.print_exc()
        if produced:
            keys_text = ", ".join(f"{k} (n={n})" for k, n in produced.items())
            print(f"Conduction velocity maps produced: {keys_text}")
        else:
            print(
                "No CV maps produced. Run Compute all (or step through points) "
                "first so LAT anchors are available."
            )


# Import order pattern:
# 1) standard/third-party libraries, 2) local project modules.
# This makes dependencies predictable and easier to scan.
# public libs
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageGrab
import pandas as pd
import librosa
from tkinter import filedialog
from tkinter import messagebox
import threading
import time
import traceback


# private libs
from .table_pkg import TableWidget
from .carto import Carto
from .ui import Toplevel as tp
from . import patch_global
from .controller import AppController
from .plotting import PlotPresenter
from .session_store import SessionStore
from .mediators.table_mediator import TableAppGlue
from .mediators.mesh_mediator import MeshAppGlue
from .viewer3d import CartoMeshPanel
from .ml_inference import (
    MLBundle,
    extract_features_from_delta_entry,
    find_local_model_files,
    load_ml_bundle,
    predicted_label_for_delta,
)

# Auto-reject threshold: signals where the SR window (voltage_sinus[0]) OR
# the first stim (voltage_stim[0]) has a peak-to-peak voltage below this
# floor are labelled "Reject" before prediction. Later stims (S2, S3) are
# not part of the gate — only the SR and S1 amplitudes matter for
# rejecting noisy / non-capturing acquisitions. The reject filter in
# MeshAppGlue.get_delta_values_for then excludes them from every
# interpolation pass.
ML_PEAK_TO_PEAK_MIN_MV = 0.1



class App(tk.Tk):
    # Class attributes (shared state):
    # these values live on the class, so all App instances see the same defaults unless overridden.
    # Here lists are used so values can be mutated in-place from settings windows.
    apps=[]
    n_fft=[100]
    hop_length=[5]
    win_length=[35]
    high_b0=[40]
    high_b1=[200]
    low_b0=[3]
    low_b1=[150]
    len_hann=[5]
    max_pooling_length=[1]
    TH=[0.45]
    # Dependency slots can be overridden by child app variants without changing core logic.
    controller_cls = AppController
    presenter_cls = PlotPresenter
    store_cls = SessionStore
    table_glue_cls = TableAppGlue
    mesh_glue_cls = MeshAppGlue
    def __init__(self, name="Saman", carto:Carto=None):
        # __init__ is the object constructor: runs once when App(...) is created.
        # `name="Saman"` gives a default argument.
        # `carto: Carto` is a type hint for readability/editor assistance.
        # `=None` means the parameter is optional at call time.
        self.forcefull=False
        self.apps.append(self)
        self.is_running=True
        self.carto = carto
        self.cont = self.carto.cont
        self.triple_active=False
        self.name=name
        self.VT_active=False
        self.check_boxes={}
        self.librosa = librosa
        # Composition: this class owns helper objects and delegates specialized work to them.
        # This keeps the main window class smaller and easier to maintain.
        self.session_store = self.store_cls()
        self.controller = self.controller_cls(self)
        self.plot_presenter = self.presenter_cls(self)
        # Mesh re-interp must run AFTER the plot has actually finished writing
        # to ``self.delta`` (otherwise it would interpolate against stale
        # values). The async triple-extra path makes this listener mandatory.
        try:
            self.plot_presenter.add_plot_done_listener(self._on_plot_done)
        except Exception:
            traceback.print_exc()
        self.table_glue = self.table_glue_cls(self)
        self.mesh_glue = self.mesh_glue_cls(self)
        self.mesh_panel = None
        self.i, self.j = 0, 0
        self._compute_all_running = False
        self._compute_all_queue: list[int] = []
        self._compute_all_saved_ij = (0, 0)
        self.Table=[]
        self.i_j_to_index(labels="hide")
        self.creating_delta()
        # going to the first item of the list, which is arbitrary and could be any other numbers, 
        # and accesss the first element that is a dataframe containing information of the points
        all_columns=self.carto.cont[0][0].columns
        self.Table=pd.DataFrame(self.Table,columns=all_columns)
        self.Table=pd.concat([self.Table,pd.DataFrame(np.zeros(len(self.to_i_j)),columns=["Coment"])],axis=1)
        # Prediction column sits BEFORE ``delta`` so the presenter's existing
        # ``vals[-1] = delta_summary`` write keeps pointing at the right cell.
        self.Table=pd.concat([self.Table,pd.DataFrame(np.full(len(self.to_i_j), "", dtype=object),columns=["Prediction"])],axis=1)
        self.Table=pd.concat([self.Table,pd.DataFrame(np.zeros(len(self.to_i_j)),columns=["delta"])],axis=1)
        # ML state. ``ml_bundle`` is loaded lazily from a joblib file (see
        # ``model/ml_model_io.py``); ``ml_selected_models`` is the list of
        # bundle entries currently used for prediction. Empty list => off.
        self.ml_bundle: MLBundle | None = None
        self.ml_selected_models: list[str] = []

    def creating_delta(self):
        #delta is a list of dictionaries that contains the information about the stimulations and sinus signals.
        #preallocation of the delta list
        self.delta=[0]*len(self.to_i_j)
        
    def i_j_to_index(self,labels="unhide"):
        self.to_index=[]
        self.to_i_j=[]
        self.labels_memory=[]
        carto=self.carto
        ind=0
        # structure of the cont: [section1,section2,section3,...]
        # structure of the section: [dataframe, name of the file, signals]
        # enumerate(iterable) yields both index and value in one loop:
        # i = numeric position, section = current element.
        for i,section in enumerate(carto.cont):
            self.to_index.append([])
            self.labels_memory.append([])
            section: tuple[pd.DataFrame, str, pd.DataFrame]
            # Nested enumerate creates a 2D traversal (section index i, row index j).
            for j,dat in enumerate(section[0].values):
                self.labels_memory[i].append(carto.cont[i][0].loc[j,"label_color"])
                #hiding the labels
                if labels=="hide":
                    carto.cont[i][0].loc[j,"label_color"]=""
                # Index mapping concept:
                # 2D coordinates (section i, point j) map to one flat row index for table syncing.
                self.to_index[i].append(ind)
                # Reverse mapping lets us jump from a table row back to the original data point.
                self.to_i_j.append([i,j])
                # updating the table matrix
                self.Table.append(dat)
                # increaing the index after each nested itteration
                ind+=1
                


    def start(self):

        # super() calls parent class behavior (tk.Tk initialization) before custom setup.
        super().__init__()
        print("start")
        self.start_x_y = []
        self.direction=1   
        self.geometry("600x600")
        self.title(f"My {self.name} APP")

        self.frame=tk.Frame(self,padx=5,pady=10,background="grey")
        self.frame.pack(fill="x",expand=False)
        self.button_dropdown=tk.Button(self.frame,text="Options",command=self.drop_down)
        self.button_dropdown.pack(side=tk.LEFT,padx=10)
        self.label = tk.Label(self.frame, text=f"point {self.cont[self.i][0]['point number'].values[self.j]}",bg="grey",fg="white")
        self.label.config(font=("timesnewroman", 10))
        self.label.pack(fill="x",side="left",padx=10)
        
        self.check_boxes={"Energy":None,"Only_Green":None}
        for key in self.check_boxes.keys():
            self.check_boxes[key]=tk.IntVar()
            check_box=tk.Checkbutton(self.frame,variable=self.check_boxes[key],command=self.checker,text=key,font=("timesnewroman",10))
            check_box.pack(side=tk.LEFT,fill="x", expand=False)
        self.button_trip=tk.Button(self.frame,text="Switch to Triple Extra Protocol",command=self.triple_protocol)
        self.button_trip.pack(side=tk.LEFT,padx=10)
        self.button_VT=tk.Button(self.frame,text="Switch to VT Protocol",command=self.VT_protocol)
        self.button_VT.pack(side=tk.LEFT,padx=10)
        # lambda creates a tiny anonymous function used as a callback.
        self.button_screen=tk.Button(self.frame,text="screen shot",command=lambda name=None:self.capture_window(name))
        self.button_screen.pack(side=tk.LEFT,padx=10)
        self.button_compute_all = tk.Button(
            self.frame,
            text="Compute all",
            command=self._compute_all_clicked,
            font=("timesnewroman", 10),
        )
        self.button_compute_all.pack(side=tk.LEFT, padx=10)
        self.button_global_patch = tk.Button(
            self.frame,
            text="Compute patch dv/ds",
            command=self._compute_global_patch_clicked,
            font=("timesnewroman", 10),
        )
        self.button_global_patch.pack(side=tk.LEFT, padx=10)
        # ML toolbar row: model dropdown + loader + predicted-label display.
        # Lives in its own frame so the main toolbar above doesn't overflow
        # horizontally when many models / long names are listed.
        self.frame_ml = tk.Frame(self, padx=5, pady=4, background="grey")
        self.frame_ml.pack(fill="x", expand=False)
        tk.Label(
            self.frame_ml, text="Model:", bg="grey", fg="white",
            font=("timesnewroman", 10),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.ml_model_var = tk.StringVar(value="(none)")
        self.ml_combo = ttk.Combobox(
            self.frame_ml,
            textvariable=self.ml_model_var,
            values=["(none)"],
            state="readonly",
            width=24,
        )
        self.ml_combo.pack(side=tk.LEFT, padx=4)
        self.ml_combo.bind("<<ComboboxSelected>>", self._on_ml_model_change)
        self.button_ml_load = tk.Button(
            self.frame_ml,
            text="Load model…",
            command=self._open_ml_dialog,
            font=("timesnewroman", 10),
        )
        self.button_ml_load.pack(side=tk.LEFT, padx=4)
        self.pred_label = tk.Label(
            self.frame_ml,
            text="Predicted: —",
            bg="grey",
            fg="white",
            font=("timesnewroman", 10, "bold"),
        )
        self.pred_label.pack(side=tk.LEFT, padx=10)
        # Conduction-velocity compute is independent of ML prediction but
        # sits in the same row to group "post-processing" actions together.
        self.button_compute_cv = tk.Button(
            self.frame_ml,
            text="Compute Conduction Velocity",
            command=self._compute_conduction_velocity_clicked,
            font=("timesnewroman", 10),
        )
        self.button_compute_cv.pack(side=tk.LEFT, padx=10)
        # Auto-discover joblib bundles in the sibling ``model/`` folder so the
        # dropdown can offer them without forcing the user to file-dialog
        # every time. Selection stays at "(none)" until the user picks one.
        self._refresh_ml_discovery()
        self.panned_window=ttk.PanedWindow(self,orient="vertical")
        self.panned_window.pack(expand=True,fill=tk.BOTH)
        self.frame3=tk.Frame(self)
        self.panned_window.add(self.frame3,weight=1)
        self.table=TableWidget(self.frame3,self.Table)
        self.table.pack(fill="both", expand=True)  
        tv = self.table.tree
        tv.defaults = {1: ["Reject", "POS", "NEG"]}
        

        # Horizontal split for the lower pane: matplotlib plots on the left,
        # OpenGL Carto mesh viewer on the right. Wrapping the existing plot
        # frame in a ttk.PanedWindow lets the user resize the 3D pane without
        # touching any of the original plotting code paths below.
        self.plot_pane = ttk.PanedWindow(self.panned_window, orient="horizontal")
        self.panned_window.add(self.plot_pane, weight=2)

        self.frame1=tk.Frame(self.plot_pane,pady=5,background="white")
        self.plot_pane.add(self.frame1, weight=3)

        # 3D mesh viewer pane. Built defensively: if the Carto object can't
        # provide a mesh (no .mesh file, missing OpenGL drivers, etc.) the
        # panel falls back to a label instead of crashing the whole app.
        self.frame_mesh = tk.Frame(self.plot_pane, background="black")
        self.plot_pane.add(self.frame_mesh, weight=2)
        try:
            self.mesh_panel = CartoMeshPanel(self.frame_mesh, self.carto)
            self.mesh_panel.pack(fill="both", expand=True)
            # Hook the mediator: spheres at every table row, click <-> select.
            try:
                self.mesh_glue.attach(self.mesh_panel)
            except Exception:
                traceback.print_exc()
        except Exception as _mesh_exc:
            traceback.print_exc()
            tk.Label(
                self.frame_mesh,
                text=f"3D viewer disabled: {_mesh_exc}",
                bg="black", fg="white", wraplength=240, justify="left",
            ).pack(fill="both", expand=True, padx=8, pady=8)
            self.mesh_panel = None

        # Per-patch dv/ds state. Populated by "Compute patch dv/ds":
        #   patch_global_results = {section_i: {window_type: {dvds_patch, ...}}}
        # A "patch" is one acquisition take/section. When results exist,
        # set_figure adds a 4th "dvds" subplot under the bipolar axis; a
        # window-type selector + time slider drive the cached |dV/ds| field
        # on the 3D mesh, and the take owning the currently-navigated point
        # is the active patch (mesh + subplot follow navigation).
        self.patch_global_results: dict = {}
        self.patch_active_window: str | None = None
        self.patch_time_index: int = 0
        self._patch_slider = None
        self._patch_window_combo = None
        self._patch_window_var = None
        self._patch_preview_on = False
        self._patch_cursor_line = None
        # Lazy per-patch compute: shared mesh operators are built once and
        # reused; each take/section is computed on first visit and cached in
        # ``patch_global_results`` so re-visits are instant.
        self._patch_ops = None
        self._patch_mode_on = False
        self._patch_sections_computing: set[int] = set()

        # self.axes is a dictionary that contains the axes objects for the top, mid, and bottom axes.
        self.set_figure()
        self._build_legend_canvases()
        self.main()
        self.plot()
        

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
        
    def set_figure(self, with_dvds=False, mod=False):
        """(Re)build the matplotlib figure embedded in ``frame1``.

        Normally three stacked subplots (``top`` unipolar, ``mid`` bipolar,
        ``bot`` reference). When ``with_dvds`` is True a 4th subplot ``dvds``
        is inserted directly under the bipolar axis (order top, mid, dvds,
        bot) for the global-patch dV/ds curve. ``mod`` first clears every
        existing widget from ``frame1`` (used when rebuilding the layout).
        """
        if mod:
            for widget in self.frame1.winfo_children():
                widget.destroy()
            self._patch_slider = None
            self._patch_window_combo = None
            self._patch_cursor_line = None
            prev = getattr(self, "fig", None)
            if prev is not None:
                try:
                    plt.close(prev)
                except Exception:
                    traceback.print_exc()

        self.with_dvds = bool(with_dvds)
        keys = ["top", "mid", "dvds", "bot"] if with_dvds else ["top", "mid", "bot"]
        nrows = len(keys)

        self.fig = plt.figure()
        plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.05)
        self.fig.clf()
        self.axes = {}
        for r, k in enumerate(keys):
            self.axes[k] = self.fig.add_subplot(nrows, 1, r + 1)
        self.axes["top"].set_xlim([0,2.5]); self.axes["top"].set_ylim([-1,1])
        self.axes["mid"].set_xlim([0,2.5]); self.axes["mid"].set_ylim([-1,1])
        self.axes["bot"].set_xlim([0,2.5]); self.axes["bot"].set_ylim([-10,10])
        if with_dvds:
            self.axes["dvds"].set_xlim([0,2.5]); self.axes["dvds"].set_ylim([0,1])
        self._fig_nrows = nrows
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame1)
        self.canvas.get_tk_widget().grid(column=0,row=0,rowspan=nrows,sticky=tk.NSEW)

    def _build_legend_canvases(self):
        """(Re)create the per-axis legend canvases and grid weights.

        Driven by the current ``self.axes`` keys so it works for both the
        3-row and 4-row (dv/ds) layouts.
        """
        self.ccs = {}
        for r, key in enumerate(self.axes.keys()):
            cc = tk.Canvas(self.frame1, width=110, height=130, bg="white", highlightbackground="white")
            cc.grid(column=1, row=r, sticky="")
            self.ccs[key] = cc
        nrows = len(self.axes)
        for r in range(nrows):
            self.frame1.grid_rowconfigure(r, weight=1)
        # The control row (slider/selector) sits just below the subplots.
        self.frame1.grid_rowconfigure(nrows, weight=0)
        self.frame1.grid_columnconfigure(0, weight=6)
        self.frame1.grid_columnconfigure(1, weight=1)

        

    def _bind_canvas_events(self):
        """Connect matplotlib canvas events. Re-run whenever the canvas is
        rebuilt (e.g. when the dv/ds subplot is added)."""
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

    # --------------------------------------------------------------- global patch
    def _mesh_viewer(self):
        panel = getattr(self, "mesh_panel", None)
        if panel is None:
            return None
        return getattr(panel, "viewer", None)

    def _auto_reject_for_global_patch(self) -> int:
        """Label points with < 3 stim windows or < 1 SR window as ``Reject``.

        These signals can't contribute to a full S1/S2/S3 + SR global patch,
        so they are auto-rejected (carto label + delta + table) before the
        compute gathers anchors. Returns the number newly rejected.
        """
        n = 0
        for gidx, entry in enumerate(self.delta):
            if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
                continue
            if str(entry[1] or "").strip().lower() == "reject":
                continue
            c1 = entry[2]
            if not isinstance(c1, dict):
                continue
            n_stim, n_sr = patch_global.count_valid_windows(c1)
            if n_stim < 3 or n_sr < 1:
                try:
                    self._mark_row_reject(gidx, c1)
                    n += 1
                except Exception:
                    traceback.print_exc()
        return n

    def _compute_global_patch_clicked(self) -> None:
        """Build per-take/window |dV/ds| patch maps over all non-rejected points.

        Each acquisition take/section is treated as a patch; for every take
        and window type the take's electrodes interpolate the unipolar V over
        the take's own footprint and the spatial gradient is cached. Runs the
        heavy harmonic pipeline on a background thread, shows a progress
        window, then (on success) adds the dv/ds subplot + time slider and
        drives the cached field on the 3D mesh.
        """
        if getattr(self, "_global_patch_running", False):
            return
        viewer = self._mesh_viewer()
        if viewer is None:
            messagebox.showerror("Compute patch dv/ds", "3D mesh viewer is not available.")
            return
        if not any(isinstance(e, (list, tuple)) and len(e) >= 3 for e in self.delta):
            messagebox.showinfo(
                "Compute patch dv/ds",
                "No analysed signals found. Run 'Compute all' first so each "
                "point has its stim/SR windows populated, then try again.",
            )
            return
        try:
            if not viewer._mesh_loaded:
                viewer._load_mesh()
        except Exception:
            traceback.print_exc()

        self._global_patch_running = True
        try:
            self.button_global_patch.config(state=tk.DISABLED)
        except Exception:
            traceback.print_exc()

        win = self._open_patch_progress_window()
        start_t = time.time()

        def progress_cb(done, total, msg):
            self.after(0, lambda: self._update_patch_progress(win, done, total, msg, start_t))

        section = self._active_section()

        def worker():
            try:
                if self._patch_ops is None:
                    self.after(0, lambda: self._update_patch_progress(
                        win, 0, 1, "Building mesh operators…", start_t))
                    ops = patch_global.build_shared_ops(self)
                else:
                    ops = self._patch_ops
                res = patch_global.compute_section_patches(
                    self, section, ops=ops, progress_cb=progress_cb)
                self.after(0, lambda: self._on_first_patch_done(ops, section, res, win))
            except Exception as exc:
                traceback.print_exc()
                self.after(0, lambda e=exc: self._on_global_patch_error(e, win))

        threading.Thread(target=worker, daemon=True, name="patch-dvds").start()

    def _open_patch_progress_window(self):
        win = tk.Toplevel(self)
        win.title("Compute patch dv/ds")
        win.geometry("440x150")
        win.transient(self)
        tk.Label(
            win, text="Computing per-patch |dV/ds| maps…",
            font=("timesnewroman", 11, "bold"),
        ).pack(pady=(14, 6))
        pb = ttk.Progressbar(win, orient="horizontal", length=380, mode="determinate")
        pb.pack(pady=4)
        lbl = tk.Label(win, text="Starting…", font=("timesnewroman", 9))
        lbl.pack(pady=2)
        elapsed = tk.Label(win, text="Elapsed: 0.0 s", font=("timesnewroman", 9))
        elapsed.pack(pady=2)
        win._pb = pb
        win._lbl = lbl
        win._elapsed = elapsed
        # Block the close button while computing.
        win.protocol("WM_DELETE_WINDOW", lambda: None)
        return win

    def _update_patch_progress(self, win, done, total, msg, start_t):
        if win is None or not win.winfo_exists():
            return
        try:
            win._pb["value"] = (100.0 * done / total) if total else 0.0
            win._lbl.config(text=f"{msg}  ({done}/{total})")
            win._elapsed.config(text=f"Elapsed: {time.time() - start_t:.1f} s")
        except Exception:
            traceback.print_exc()

    def _on_global_patch_error(self, exc, win):
        self._global_patch_running = False
        try:
            self.button_global_patch.config(state=tk.NORMAL)
        except Exception:
            traceback.print_exc()
        if win is not None and win.winfo_exists():
            win.destroy()
        messagebox.showerror("Compute patch dv/ds", f"Failed: {exc}")

    def _on_first_patch_done(self, ops, section, res, win):
        """Set up the patch UI after the first take has been computed."""
        self._global_patch_running = False
        try:
            self.button_global_patch.config(state=tk.NORMAL)
        except Exception:
            traceback.print_exc()
        if win is not None and win.winfo_exists():
            win.destroy()
        self._patch_ops = ops
        if not res:
            messagebox.showinfo(
                "Compute patch dv/ds",
                "This take has no qualifying patch.\n\nA take/window patch "
                f"needs at least {patch_global._MIN_PATCH_ANCHORS} non-rejected "
                "points with a measurable window. Run 'Compute all' first so "
                "windows/references are populated, then navigate to a take "
                "with enough points.",
            )
            return

        self.patch_global_results = {int(section): res}
        # All four windows are offered; takes lacking one just show "no patch".
        order = list(patch_global.WINDOW_TYPES)
        # Prefer the first window this take actually has.
        self.patch_active_window = next(
            (w for w in order if w in res), order[0]
        )
        self.patch_time_index = 0
        self._patch_mode_on = True

        # Rebuild the figure with the 4th dv/ds subplot, then re-create the
        # legend canvases, event bindings, and the slider/selector controls.
        self.set_figure(with_dvds=True, mod=True)
        self._build_legend_canvases()
        self._bind_canvas_events()
        self._build_patch_controls(order)
        self._begin_patch_mesh_preview()
        # Redraw signals (top/mid/bot) and refresh the active take's patch
        # (mesh + dv/ds subplot). As the user navigates to other takes they
        # are computed lazily and cached.
        self.update_plot()
        self._sync_patch_to_current_point()

        wins_here = ", ".join(w for w in order if w in res)
        messagebox.showinfo(
            "Compute patch dv/ds",
            f"Take {section} computed ({wins_here}).\n\nNavigate to other "
            "points to compute their takes on demand (cached after the first "
            "visit).",
        )

    def _sync_patch_to_current_point(self):
        """Show the take owning the current point, computing it if needed."""
        if not self._patch_mode_on or "dvds" not in getattr(self, "axes", {}):
            return
        sec = self._active_section()
        if sec in self.patch_global_results:
            self._refresh_active_patch()
        else:
            self._ensure_section_computed_async(sec)

    def _ensure_section_computed_async(self, section):
        """Compute one take's patches on a worker thread, then cache+refresh."""
        sec = int(section)
        if sec in self.patch_global_results or sec in self._patch_sections_computing:
            self._refresh_active_patch()
            return
        if self._patch_ops is None:
            return
        self._patch_sections_computing.add(sec)
        # Busy hint on the dv/ds subplot while the take computes.
        try:
            ax = self.axes.get("dvds")
            if ax is not None:
                ax.clear()
                ax.set_title(f"dV/ds  [{self.patch_active_window}]  computing take {sec}…",
                             fontsize=8)
                ax.set_xlabel("time (s)", fontsize=7)
                ax.set_ylabel("|dV/ds|", fontsize=7)
                ax.set_xlim([0, 2.5])
                self.canvas.draw_idle()
        except Exception:
            traceback.print_exc()

        ops = self._patch_ops

        def worker():
            try:
                res = patch_global.compute_section_patches(self, sec, ops=ops)
                self.after(0, lambda: self._on_section_computed(sec, res))
            except Exception:
                traceback.print_exc()
                self.after(0, lambda: self._on_section_computed(sec, {}))

        threading.Thread(target=worker, daemon=True, name=f"patch-take-{sec}").start()

    def _on_section_computed(self, section, res):
        sec = int(section)
        self._patch_sections_computing.discard(sec)
        self.patch_global_results[sec] = res or {}
        # Only refresh if the user is still on this take.
        if self._active_section() == sec:
            self._refresh_active_patch()

    # ----- per-patch accessors -------------------------------------------
    def _active_section(self) -> int:
        """The take/section that owns the currently-navigated point."""
        try:
            return int(self.i)
        except Exception:
            return -1

    def _active_patch_result(self):
        """Result dict for the current take + active window, or ``None``."""
        if not self.patch_global_results:
            return None
        bywt = self.patch_global_results.get(self._active_section())
        if not bywt:
            return None
        return bywt.get(self.patch_active_window)

    def _patch_abs_time_seconds(self, res, rel=None):
        """Absolute signal time (seconds) for a patch result's sample axis.

        Uses the currently-navigated point's own reference when it is an
        anchor of this patch, else the patch's representative reference, so
        the dv/ds curve lands where the window actually sits in the 0-2.5 s
        trace.
        """
        if rel is None:
            rel = res["rel"]
        fs = float(res.get("fs") or patch_global.DEFAULT_FS) or patch_global.DEFAULT_FS
        ref = int(res.get("ref_repr", 0))
        try:
            gidx = self.to_index[self.i][self.j]
            c1 = self.delta[gidx][2]
            span = patch_global._window_ref_and_span(c1, self.patch_active_window)
            if span is not None:
                ref = int(span[0])
        except Exception:
            pass
        return (ref + np.asarray(rel, dtype=np.float64)) / fs

    def _patch_full_field(self, res, idx):
        """Scatter a patch sample into a full per-vertex array (NaN elsewhere)."""
        n_v = int(np.asarray(self.carto.vertices).shape[0])
        full = np.full(n_v, np.nan, dtype=np.float64)
        pv = np.asarray(res["patch_vertices"], dtype=np.int64)
        series = res["dvds_patch"]
        k = max(0, min(series.shape[0] - 1, int(idx)))
        full[pv] = np.asarray(series[k], dtype=np.float64)
        return full

    def _build_patch_controls(self, order):
        """Window-type selector + time slider under the subplots (in frame1)."""
        nrows = len(self.axes)
        frame = tk.Frame(self.frame1, background="white")
        frame.grid(column=0, row=nrows, columnspan=2, sticky="ew", pady=2)
        tk.Label(frame, text="Window:", background="white").pack(side=tk.LEFT, padx=(4, 2))
        self._patch_window_var = tk.StringVar(value=self.patch_active_window)
        combo = ttk.Combobox(
            frame, textvariable=self._patch_window_var, values=list(order),
            state="readonly", width=6,
        )
        combo.pack(side=tk.LEFT, padx=4)
        combo.bind("<<ComboboxSelected>>", self._on_patch_window_change)
        self._patch_window_combo = combo
        tk.Label(frame, text="Time:", background="white").pack(side=tk.LEFT, padx=(10, 2))
        res = self._active_patch_result()
        n = int(res["dvds_patch"].shape[0]) if res is not None else 1
        self._patch_slider = ttk.Scale(
            frame, from_=0, to=max(0, n - 1), orient="horizontal",
            command=self._on_patch_slider_change,
        )
        self._patch_slider.pack(side=tk.LEFT, fill="x", expand=True, padx=6)
        self._patch_time_label = tk.Label(frame, text="", background="white", width=12)
        self._patch_time_label.pack(side=tk.LEFT, padx=6)
        self._update_patch_time_label()

    def _begin_patch_mesh_preview(self):
        viewer = self._mesh_viewer()
        if viewer is None or not self.patch_global_results:
            return
        try:
            if viewer.begin_patch_preview(f"patch:dvds_{self.patch_active_window}"):
                self._patch_preview_on = True
        except Exception:
            traceback.print_exc()

    def _push_patch_field_to_mesh(self):
        viewer = self._mesh_viewer()
        res = self._active_patch_result()
        if viewer is None or not self._patch_preview_on:
            return
        try:
            if res is None:
                # Current take has no patch for this window: blank the mesh.
                n_v = int(np.asarray(self.carto.vertices).shape[0])
                viewer.set_patch_preview_field(np.full(n_v, np.nan, dtype=np.float64))
                return
            viewer.set_patch_preview_field(self._patch_full_field(res, self.patch_time_index))
        except Exception:
            traceback.print_exc()

    def _refresh_active_patch(self):
        """Re-sync slider range, mesh field, colour range + subplot for the
        take that owns the current point. Called after compute, on window
        change, and whenever navigation lands on a different take."""
        if not self.patch_global_results or "dvds" not in getattr(self, "axes", {}):
            return
        res = self._active_patch_result()
        if res is not None:
            n = int(res["dvds_patch"].shape[0])
            self.patch_time_index = max(0, min(n - 1, int(self.patch_time_index)))
            if self._patch_slider is not None:
                try:
                    self._patch_slider.configure(to=max(0, n - 1))
                    self._patch_slider.set(self.patch_time_index)
                except Exception:
                    traceback.print_exc()
            viewer = self._mesh_viewer()
            if viewer is not None and self._patch_preview_on:
                try:
                    viewer.set_patch_preview_color_range(res["vmin"], res["vmax"])
                except Exception:
                    traceback.print_exc()
        self._push_patch_field_to_mesh()
        self._update_patch_time_label()
        self._redraw_patch_subplot()

    def _on_patch_slider_change(self, value):
        if not self.patch_global_results or self.patch_active_window is None:
            return
        try:
            idx = int(round(float(value)))
        except (TypeError, ValueError):
            return
        res = self._active_patch_result()
        if res is None:
            return
        n = int(res["dvds_patch"].shape[0])
        self.patch_time_index = max(0, min(n - 1, idx))
        self._push_patch_field_to_mesh()
        self._update_patch_time_label()
        self._update_patch_cursor()

    def _on_patch_window_change(self, _event=None):
        if self._patch_window_var is None:
            return
        wt = self._patch_window_var.get()
        if wt not in patch_global.WINDOW_TYPES:
            return
        self.patch_active_window = wt
        self.patch_time_index = 0
        viewer = self._mesh_viewer()
        if viewer is not None and self._patch_preview_on:
            try:
                viewer.set_patch_preview_label(f"patch:dvds_{wt}")
            except Exception:
                traceback.print_exc()
        self._refresh_active_patch()

    def _update_patch_time_label(self):
        res = self._active_patch_result()
        lbl = getattr(self, "_patch_time_label", None)
        if lbl is None:
            return
        if res is None:
            try:
                lbl.config(text="—")
            except Exception:
                traceback.print_exc()
            return
        t_abs = self._patch_abs_time_seconds(res)
        if 0 <= self.patch_time_index < t_abs.size:
            try:
                lbl.config(text=f"{t_abs[self.patch_time_index]:.3f} s")
            except Exception:
                traceback.print_exc()

    def _redraw_patch_subplot(self):
        """Draw the current point's patch dv/ds vs absolute time (0-2.5 s)."""
        if not self.patch_global_results or "dvds" not in getattr(self, "axes", {}):
            return
        ax = self.axes["dvds"]
        ax.clear()
        self._patch_cursor_line = None
        res = self._active_patch_result()
        title = f"dV/ds  [{self.patch_active_window}]"
        if res is None:
            ax.set_title(title + "  (no patch for this take/window)", fontsize=8)
            ax.set_xlabel("time (s)", fontsize=7)
            ax.set_ylabel("|dV/ds|", fontsize=7)
            ax.set_xlim([0, 2.5])
            self.canvas.draw_idle()
            return
        t_abs = self._patch_abs_time_seconds(res)
        try:
            gidx = self.to_index[self.i][self.j]
        except Exception:
            gidx = -1
        vidx = res["anchor_vertex"].get(int(gidx))
        col = None
        if vidx is not None:
            pv = np.asarray(res["patch_vertices"], dtype=np.int64)
            pos = int(np.searchsorted(pv, int(vidx)))
            if 0 <= pos < pv.size and int(pv[pos]) == int(vidx):
                col = pos
        if col is not None and t_abs.size:
            series = res["dvds_patch"][:, col]
            ax.plot(t_abs, series, color="tab:purple", linewidth=1.0)
            title += f"  point #{gidx}"
        else:
            title += "  (current point not an anchor for this take/window)"
        if 0 <= self.patch_time_index < t_abs.size:
            self._patch_cursor_line = ax.axvline(
                float(t_abs[self.patch_time_index]), color="red", linewidth=1.0
            )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_ylabel("|dV/ds|", fontsize=7)
        ax.set_xlim([0, 2.5])
        self.canvas.draw_idle()

    def _update_patch_cursor(self):
        res = self._active_patch_result()
        if res is None or "dvds" not in getattr(self, "axes", {}):
            return
        t_abs = self._patch_abs_time_seconds(res)
        if not (0 <= self.patch_time_index < t_abs.size):
            return
        x = float(t_abs[self.patch_time_index])
        try:
            if self._patch_cursor_line is not None:
                self._patch_cursor_line.set_xdata([x, x])
            else:
                self._patch_cursor_line = self.axes["dvds"].axvline(x, color="red", linewidth=1.0)
            self.canvas.draw_idle()
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
        # Cheap UI sync first so the click feels instant even on big sessions.
        for i in self.axes.values():
            i.clear()
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
        # Re-sync the patch (mesh + slider + dv/ds subplot) to the take that
        # owns the new point, computing it lazily if not cached. The subplot
        # was cleared by the loop above.
        if self._patch_mode_on and "dvds" in self.axes:
            try:
                self._sync_patch_to_current_point()
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
            self.session_store.save_delta(path=file_path, delta=self.delta)
            print(f"File is saved in {file_path}")

    def _delta_label_from_entry(self, index, delt):
        try:
            i, j = self.to_i_j[index]
            point_number = self.carto.cont[i][0].loc[j, "point number"]
            print(delt[0], delt[1], point_number)
            if point_number == delt[0]:
                self.carto.cont[i][0].loc[j, "label_color"] = delt[1]
                return delt[1]
            print(f"mismatch{point_number} {delt[0]}")
            return "mismatch"
        except Exception as e:
            print(e)
            traceback.print_exc()
            return ""

    def _delta_summary_from_entry(self, delt):
        try:
            summary = ", ".join(
                [
                    f"{key}: {', '.join([str(ii) for ii in value])}"
                    for key, value in delt[2].items()
                    if "voltage" not in key
                ]
            )
            return [summary]
        except Exception:
            traceback.print_exc()
            return ""

    def _refresh_table_from_delta(self):
        in_table = []
        labels = []
        for index, delt in enumerate(self.delta):
            labels.append(self._delta_label_from_entry(index, delt))
            in_table.append(self._delta_summary_from_entry(delt))

        self.Table["delta"] = pd.Series(in_table)
        self.Table["label_color"] = pd.Series(labels)
        self.table.tree.init_from_df(self.Table)
        # Re-label everything: ``init_from_df`` rebuilds the tree from
        # ``self.Table``, so the Prediction column needs to be repopulated.
        try:
            self._refresh_all_predictions()
        except Exception:
            traceback.print_exc()

    def _load_delta_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not file_path:
            return
        print(f"Data loaded from {file_path}")
        self.delta = self.session_store.load_delta(path=file_path)
        self._refresh_table_from_delta()
        try:
            if getattr(self, "mesh_glue", None) is not None:
                self.mesh_glue.notify_delta_changed()
        except Exception:
            traceback.print_exc()

    def _open_meta_parameters_window(self):
        top_window = tk.Toplevel()
        top_window.attributes("-topmost", True)
        tp(top_window, self)

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
        """Remove cached POS probabilities so the mesh map stops using them."""
        changed = False
        for entry in self.delta:
            if isinstance(entry, list) and len(entry) >= 3 and isinstance(entry[2], dict):
                if entry[2].pop("pos_proba", None) is not None:
                    changed = True
        if changed:
            try:
                if getattr(self, "mesh_glue", None) is not None:
                    self.mesh_glue.notify_delta_changed()
            except Exception:
                traceback.print_exc()

    def _clear_prediction_column(self) -> None:
        try:
            col_loc = self.Table.columns.get_loc("Prediction")
        except Exception:
            return
        self.Table.iloc[:, col_loc] = ""
        try:
            tv = self.table.tree
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

    @staticmethod
    def _min_peak_to_peak_mv(c1) -> float | None:
        """Return min(SR, S1) peak-to-peak voltage, or None if neither exists.

        The gate only looks at the SR window (``voltage_sinus[0]``) and the
        first stim (``voltage_stim[0]``). ``False`` markers (presenter's
        "no measurable window" flag) are skipped so the gate doesn't fire
        on legitimately-missing data; if both are missing we return None
        and the caller leaves the row alone.
        """
        if not isinstance(c1, dict):
            return None
        vals: list[float] = []
        for key in ("voltage_sinus", "voltage_stim"):
            arr = c1.get(key) or []
            if not arr:
                continue
            v = arr[0]
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return min(vals) if vals else None

    def _mark_row_reject(self, idx: int, c1) -> None:
        """Force ``Reject`` on carto label, delta, table cell. Idempotent."""
        try:
            i, j = self.to_i_j[idx]
            df = self.carto.cont[i][0]
            df.iat[j, df.columns.get_loc("label_color")] = "Reject"
        except Exception:
            traceback.print_exc()
        try:
            entry = self.delta[idx]
            if isinstance(entry, list) and len(entry) >= 2:
                entry[1] = "Reject"
        except Exception:
            traceback.print_exc()
        if isinstance(c1, dict):
            # Drop any previous probability so the POS/NEG map stops
            # rendering this electrode as an anchor.
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
        """Compute, store, and display the prediction for one table row.

        Called from the presenter's persist step (so navigation and Compute
        all both populate predictions) and from ``_refresh_all_predictions``
        (so changing the selected model re-labels every row in one pass).

        Two gates run before the model is invoked:

        1. Voltage-based reject: if ``min(voltage_sinus[0], voltage_stim[0])``
           (SR and S1 peak-to-peak) is below ``ML_PEAK_TO_PEAK_MIN_MV``
           (0.1 mV), the row is forced to ``Reject`` on the carto label,
           the delta entry, and the table. The mesh provider's existing
           ``reject`` filter then excludes the point from every
           interpolation pass.
        2. Feature-based reject (training-time gate, < 0.05 mV stim min):
           the feature extractor returns ``None`` and the predictor writes
           an empty cell — the point is simply not predictable.

        ``notify_mesh`` should be ``False`` when many rows are being
        updated back-to-back; the caller is then responsible for one
        final ``mesh_glue.notify_delta_changed()`` to refresh the field
        dropdown and re-interpolate the map.
        """
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
        min_pp = self._min_peak_to_peak_mv(c1)
        if min_pp is not None and min_pp < ML_PEAK_TO_PEAK_MIN_MV:
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
                # Clamp to [0, 1] — probability strategy could overshoot
                # very slightly under numerical noise.
                c1["pos_proba"] = float(min(1.0, max(0.0, proba)))
        self._write_prediction_cell(idx, label)
        if notify_mesh:
            # The POS/NEG probability anchor for this electrode just
            # changed (added, updated, or removed) — refresh the map.
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


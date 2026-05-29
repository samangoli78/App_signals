"""OpenGL mesh view: normals + lighting + 1D colormap LUT, with electrode
spheres, GPU back-buffer color picking, a live cursor tooltip and labelled
colorbar ticks.
"""

from __future__ import annotations

import copy
import ctypes
import math
import threading
import time as _time
import tkinter as tk
import traceback

import numpy as np

from . import colormap as cm
from . import geometry as geom
from . import picking as pk
from . import laplacian as lap
from .projection import project_points_to_mesh
from .text import TextRenderer

try:
    from pyopengltk import OpenGLFrame
    from OpenGL.GL import (
        GL_AMBIENT,
        GL_AMBIENT_AND_DIFFUSE,
        GL_BLEND,
        GL_CLAMP_TO_EDGE,
        GL_COLOR_ARRAY,
        GL_COLOR_BUFFER_BIT,
        GL_COLOR_MATERIAL,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_DIFFUSE,
        GL_FLOAT,
        GL_FRONT_AND_BACK,
        GL_LIGHT0,
        GL_LIGHTING,
        GL_LINE_SMOOTH,
        GL_LINES,
        GL_LINEAR,
        GL_MODELVIEW,
        GL_MODULATE,
        GL_NEAREST,
        GL_NORMALIZE,
        GL_NORMAL_ARRAY,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_POINTS,
        GL_POLYGON_OFFSET_FILL,
        GL_POSITION,
        GL_PROJECTION,
        GL_QUADS,
        GL_RGB,
        GL_SMOOTH,
        GL_SPECULAR,
        GL_SRC_ALPHA,
        GL_TEXTURE_1D,
        GL_TEXTURE_2D,
        GL_TEXTURE_COORD_ARRAY,
        GL_TEXTURE_ENV,
        GL_TEXTURE_ENV_MODE,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TRIANGLES,
        GL_UNPACK_ALIGNMENT,
        GL_UNSIGNED_BYTE,
        GL_UNSIGNED_INT,
        GL_VERTEX_ARRAY,
        GL_SHININESS,
        glBegin,
        glBindTexture,
        glBlendFunc,
        glClear,
        glClearColor,
        glColor3f,
        glColor3ub,
        glColor4f,
        glColorMaterial,
        glColorPointer,
        glDisable,
        glDisableClientState,
        glDrawArrays,
        glDrawElements,
        glEnable,
        glEnableClientState,
        glEnd,
        glFinish,
        glFlush,
        glGenTextures,
        glLightfv,
        glLineWidth,
        glLoadIdentity,
        glMaterialf,
        glMaterialfv,
        glMatrixMode,
        glNormalPointer,
        glOrtho,
        glPixelStorei,
        glPointSize,
        glPolygonOffset,
        glPopMatrix,
        glPushMatrix,
        glReadBuffer,
        glReadPixels,
        glRotatef,
        glShadeModel,
        glTexCoordPointer,
        glTexEnvi,
        glTexImage1D,
        glTexParameteri,
        glTranslatef,
        glVertex2f,
        glVertex3fv,
        glVertexPointer,
        glViewport,
    )
    from OpenGL.GL import GL_BACK  # used by glReadBuffer
    from OpenGL.GL import glGetDoublev, GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT
    from OpenGL.GLU import gluLookAt, gluPerspective, gluProject

    _GL_AVAILABLE = True
    _GL_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover
    OpenGLFrame = tk.Frame
    _GL_AVAILABLE = False
    _GL_IMPORT_ERROR = _exc


# Sphere geometry shared by every viewer instance (cheap to keep as singleton).
if _GL_AVAILABLE:
    _SPH_V, _SPH_T = geom.icosphere(subdivisions=1)
else:  # pragma: no cover
    _SPH_V = np.zeros((0, 3), dtype=np.float32)
    _SPH_T = np.zeros((0, 3), dtype=np.int32)


class CartoMeshViewer(OpenGLFrame):
    """Carto mesh + GL colorbar + electrode spheres with GPU id-picking."""

    _CB_THICK = 22
    _CB_PAD = 8
    _LUT_WIDTH = 1024
    _SPHERE_RADIUS_FRAC = 0.012  # in normalized mesh units (mesh fits in unit sphere)
    _HIGHLIGHT_SCALE = 1.6
    _BG_RGB = (0.05, 0.06, 0.09, 1.0)

    def __init__(
        self,
        master,
        carto,
        scalar_field: str = "bipolar",
        cmap_name: str = "jet",
        mesh_name: str | None = None,
        **kw,
    ) -> None:
        super().__init__(master, **kw)
        self.carto = carto
        self.mesh_name = mesh_name or getattr(carto, "name", "") or "Mesh"
        self.scalar_field = scalar_field if scalar_field in cm.SCALAR_FIELDS else "bipolar"

        self.cmap_name = cmap_name
        self.reverse_cmap = False
        self.n_bins = 256
        self.auto_range = True
        self.vmin: float | None = None
        self.vmax: float | None = None

        self.color_mode: str = "standard"
        self.piece_knots: list[float] = []
        self.custom_bins: list[dict] = []

        self.show_colorbar = True
        self.cb_orientation: str = "vertical"
        self.cb_drag_dx = 0
        self.cb_drag_dy = 0

        # Color sampled for NaN / out-of-range vertices (border of 1D LUT).
        # Mid grey reads clearly as "no scalar" while keeping full triangles
        # on screen (no index-buffer drops).
        self.no_data_color: tuple[float, float, float, float] = (0.38, 0.40, 0.44, 1.0)

        # Mesh data
        self._verts_flat: np.ndarray | None = None
        self._normals_flat: np.ndarray | None = None
        self._texcoords_flat: np.ndarray | None = None
        self._indices: np.ndarray | None = None
        # ``_draw_indices`` mirrors ``_indices``; the GS mesh path draws the
        # full index list so NaN-only triangles still receive a solid no-data colour.
        self._draw_indices: np.ndarray | None = None
        # Mesh geometry in **mesh-frame** (raw) coords, cached for diagnostics.
        self._mesh_verts_raw: np.ndarray | None = None
        self._mesh_tris_raw: np.ndarray | None = None
        self._scalars: np.ndarray | None = None
        self._mesh_loaded = False
        self._mesh_center = np.zeros(3, dtype=np.float32)
        self._mesh_radius = 1.0

        self._mesh_gl_prog: int | None = None
        self._mesh_gl_prog_ok: bool | None = None
        self._mesh_gl_loc_vert = -1
        self._mesh_gl_loc_norm = -1
        self._mesh_gl_loc_st = -1
        self._mesh_gl_u_nodata = -1
        self._mesh_gl_u_cmap = -1
        self._mesh_vbo_pos = 0
        self._mesh_vbo_norm = 0
        self._mesh_vbo_attr = 0
        self._mesh_ebo = 0
        self._mesh_vao = 0
        self._mesh_gl_nverts = 0
        # When True, use classic ``glTexCoordPointer`` rasterisation (per-pixel
        # linear blend of LUT coords). When False, use the optional GS+FS path
        # that matches Gouraud for three finite corners and partitions NaN
        # regions by barycentric closest-vertex.
        self.prefer_legacy_mesh_rendering: bool = True

        # Pick pass for mesh (per-triangle unwrapped vertex arrays).
        self._pick_mesh_verts: np.ndarray | None = None
        self._pick_mesh_colors: np.ndarray | None = None

        # Electrodes / spheres.
        self._elec_xyz_raw: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._elec_xyz_norm: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self._elec_xyz_proj_norm: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self._elec_global_idx: list[int] = []
        self._elec_labels: list[str] = []
        self._selected_global_idx: int | None = None
        self._hover_global_idx: int | None = None
        self._hover_triangle: int | None = None
        self._hover_kind: str = "empty"

        # Sphere display state. Default to surface-projected positions so the
        # spheres actually sit on the mesh even when the raw electrode coords
        # are recorded a few millimeters off the surface.
        self.sphere_radius_factor: float = 1.0
        self.use_projected: bool = True
        self._sphere_radius = self._SPHERE_RADIUS_FRAC

        # Harmonic interpolation of "delta:<metric>" virtual scalar fields.
        # The Laplacian + edge graph are built lazily on first use.
        self.interpolation_enabled: bool = False
        self.interpolation_radius: float | None = None
        self._delta_provider = None
        self._current_delta_metric: str | None = None
        self._L: "object | None" = None
        self._mesh_graph: "object | None" = None
        self._anchor_vidx: np.ndarray | None = None
        self._fields_listeners: list = []
        # Conduction-velocity fields are derived from interpolated LAT and
        # only exist after the user triggers compute_conduction_velocity().
        # Stored as ``{metric_key: per_vertex_array}`` and exposed through
        # available_fields() as ``delta:cv_*``.
        self._cv_fields: dict[str, np.ndarray] = {}

        # Geodesic distance cache: filled once per (anchor set, radius) combo
        # and reused for every metric / value update. The matrix is anchor-only
        # so it's anchors × n_vertices floats (cheap as long as we cap radius).
        self._anchor_dist: np.ndarray | None = None  # (n_anchors, n_vertices)
        # ``None`` = no cache. ``math.inf`` = uncapped geodesic field (any UI r
        # is applied later inside :func:`harmonic_interpolate_bounded_cached`).
        # Finite = capped matrix from a fallback solver (exact ``r`` match).
        self._anchor_dist_radius: float | None = None
        self._anchor_dist_anchors: tuple | None = None

        # Mean edge length of the current mesh (used to pick a sensible default
        # interpolation radius = 10× mean edge so localized solves are fast).
        self._mean_edge: float = 0.0

        # When True, every delta recompute uses the geodesic-ball cut but solves
        # one harmonic system over the whole patch (Neumann at the rim). When
        # False, the bounded cached/local path is used. The toolbar Global
        # checkbox mirrors this flag.
        self.use_global_patch_harmonic: bool = False

        # Heavy delta interpolation runs in a single dedicated worker thread
        # so Tk + GL stays responsive. The "request slot" only ever holds the
        # most recent request — newer requests overwrite older ones, so a
        # rapid sequence of clicks/selections never piles up a queue.
        #
        # ``_interp_serial`` is incremented per request and copied into the
        # request snapshot. The worker drops stale work as soon as the serial
        # no longer matches ``self._interp_serial`` (the latest posted) and the
        # main-thread ``apply`` callback rejects stale results.
        self._interp_cond = threading.Condition()
        self._interp_pending: dict | None = None
        self._interp_invalidate: bool = False
        self._interp_stop: bool = False
        self._interp_thread: threading.Thread | None = None
        self._interp_serial = 0
        self._interp_applied_serial = 0

        # Acquisition patch tool drives the main mesh with a temporary scalar field.
        self._patch_preview_active: bool = False
        self._patch_preview_snap: dict | None = None
        self._patch_vector_bases: np.ndarray | None = None
        self._patch_vector_dirs: np.ndarray | None = None
        self._patch_vector_mags: np.ndarray | None = None
        self._patch_vector_scale: float = 1.0
        self._patch_vector_mag_max: float = 1.0
        self._patch_anchor_idx: np.ndarray | None = None
        self._patch_anchor_labels: list[str] = []
        self._selected_patch_anchor: int | None = None
        self._hover_patch_anchor: int | None = None

        # LUT texture for mesh shading.
        self._tex_id: int | None = None
        self._lut_nearest = False

        self._color_listeners: list = []

        # Camera state.
        self._rot_x = 20.0
        self._rot_y = -30.0
        self._zoom = 2.8
        self._pan = [0.0, 0.0]
        self._drag_origin: tuple[int, int] | None = None
        self._drag_mode: str | None = None
        self._cb_rect: tuple[float, float, float, float] | None = None

        # Picking / hover state.
        self._pick_request: tuple[int, int] | None = None
        self._pending_click: bool = False
        self._cursor_pos: tuple[int, int] | None = None
        self._motion_after: str | None = None
        self._redraw_after_id: str | None = None
        # Timestamp + cursor position of the last successful pick. Used to
        # skip the synchronous pick render on click when the cursor is still
        # parked over the same spot as the last hover pick.
        self._last_pick_at: float = 0.0
        self._last_pick_xy: tuple[int, int] | None = None

        # Callback: ``cb(kind, payload, info)`` where kind is "sphere", "triangle"
        # or "empty"; payload is the global electrode index (sphere) or the
        # triangle index (triangle).
        self.on_pick_callback = None

        # Text renderer is allocated on first GL draw (needs a context).
        self._text: TextRenderer | None = None

        if not _GL_AVAILABLE:
            self._show_fallback_message(f"OpenGL viewer unavailable:\n{_GL_IMPORT_ERROR}")
            return

        self.bind("<ButtonPress-1>", self._on_b1_press)
        self.bind("<B1-Motion>", self._on_b1_motion)
        self.bind("<ButtonRelease-1>", self._on_b1_release)
        self.bind("<ButtonPress-3>", self._on_b3_press)
        self.bind("<B3-Motion>", self._on_b3_motion)
        self.bind("<ButtonRelease-3>", self._on_b3_release)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", self._on_wheel_linux)
        self.bind("<Button-5>", self._on_wheel_linux)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)

        self._ctx_menu = tk.Menu(self, tearoff=False)
        self._ctx_menu.add_command(label="Modify settings...", command=self.open_colorbar_settings)
        om = tk.Menu(self._ctx_menu, tearoff=False)
        om.add_command(label="Horizontal", command=lambda: self._set_cb_orient("horizontal"))
        om.add_command(label="Vertical", command=lambda: self._set_cb_orient("vertical"))
        self._ctx_menu.add_cascade(label="Orientation", menu=om)
        self._ctx_menu.add_separator()
        self._ctx_menu.add_command(label="Reset colorbar position", command=self._reset_cb_position)
        self._ctx_menu.add_command(label="Hide colorbar", command=lambda: self._toggle_cb_visible(False))

        # Right-click on an electrode sphere opens this menu.
        self._sphere_ctx_menu = tk.Menu(self, tearoff=False)
        self._sphere_ctx_menu.add_command(label="Modify spheres...", command=self.open_sphere_settings)
        self._sphere_ctx_menu.add_separator()
        self._sphere_ctx_menu.add_command(label="Bigger spheres", command=lambda: self._bump_radius_factor(1.25))
        self._sphere_ctx_menu.add_command(label="Smaller spheres", command=lambda: self._bump_radius_factor(1.0 / 1.25))
        self._sphere_ctx_menu.add_separator()
        self._sphere_ctx_menu.add_command(label="Toggle projected/original", command=self.toggle_projection)
        self._sphere_ctx_menu.add_command(label="Re-project to mesh", command=self.recompute_projection)

    # ---------------------------------------------------------------- helpers
    def _toggle_cb_visible(self, show: bool) -> None:
        self.show_colorbar = show
        self._notify_color_listeners()
        self._request_redraw()

    def _set_cb_orient(self, o: str) -> None:
        if o in ("horizontal", "vertical"):
            self.cb_orientation = o
            self._request_redraw()

    def _reset_cb_position(self) -> None:
        self.cb_drag_dx = self.cb_drag_dy = 0
        self._request_redraw()

    def _show_fallback_message(self, msg: str) -> None:
        for child in self.winfo_children():
            child.destroy()
        tk.Label(self, text=msg, bg="black", fg="white", wraplength=250, justify="left").pack(
            fill="both", expand=True, padx=8, pady=8
        )

    def _resolved_range(self) -> tuple[float, float]:
        scalars = self._scalars
        if scalars is None or scalars.size == 0:
            return 0.0, 1.0
        if not self.auto_range and self.vmin is not None and self.vmax is not None:
            return float(self.vmin), float(self.vmax)
        finite = np.isfinite(scalars)
        if not finite.any():
            return 0.0, 1.0
        valid = scalars[finite]
        smin, smax = np.percentile(valid, [2.0, 98.0])
        return float(smin), float(smax)

    def _effective_cmap(self):
        name = cm.effective_cmap_name(self.cmap_name, self.reverse_cmap)
        try:
            return cm.get_cmap(name)
        except Exception:
            return cm.get_cmap("jet")

    def _scalar_values_to_rgb(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        vmin, vmax = self._resolved_range()
        if self.color_mode == "custom" and self.custom_bins:
            out = np.zeros((v.size, 3), dtype=np.float32)
            bins = sorted(self.custom_bins, key=lambda b: float(b["lo"]))
            for b in bins:
                m = (v >= float(b["lo"])) & (v <= float(b["hi"]))
                rgb = np.asarray(b["rgb"], dtype=np.float32).reshape(3)
                out[m] = rgb
            if bins:
                m_left = v < float(bins[0]["lo"])
                m_right = v > float(bins[-1]["hi"])
                out[m_left] = np.asarray(bins[0]["rgb"], dtype=np.float32)
                out[m_right] = np.asarray(bins[-1]["rgb"], dtype=np.float32)
            return out

        if vmax - vmin < 1e-12:
            t = np.zeros_like(v)
        else:
            t = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
        t = np.where(np.isfinite(t), t, 0.5)
        u = cm.t_to_cmap_u(t, self.piece_knots)
        cmap = self._effective_cmap()
        return cmap(u)[:, :3].astype(np.float32)

    def _ensure_texture(self) -> None:
        if self._tex_id is not None:
            return
        tid = glGenTextures(1)
        if isinstance(tid, (list, tuple, np.ndarray)):
            self._tex_id = int(np.asarray(tid).ravel()[0])
        else:
            self._tex_id = int(tid)

    def _upload_lut_texture(self) -> None:
        if not _GL_AVAILABLE or self._scalars is None:
            return
        vmin, vmax = self._resolved_range()
        rgb, nearest = cm.build_1d_lut_rgb(
            lut_width=self._LUT_WIDTH,
            color_mode=self.color_mode,
            cmap_name=self.cmap_name,
            reverse_cmap=self.reverse_cmap,
            n_bins=self.n_bins,
            piece_knots=self.piece_knots,
            custom_bins=self.custom_bins,
            vmin=vmin,
            vmax=vmax,
        )
        self._lut_nearest = nearest
        self._ensure_texture()
        glBindTexture(GL_TEXTURE_1D, self._tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        w = rgb.shape[0]
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, w, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.tobytes())
        filt = GL_NEAREST if nearest else GL_LINEAR
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, filt)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, filt)
        # NaN/no-data vertices carry texcoord == colormap.NAN_TEXCOORD (-1).
        # Set the wrap mode to CLAMP_TO_BORDER + a dark muted border colour so
        # those vertices render as "no data" instead of getting a misleading
        # mid-LUT colour.
        try:
            from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_TEXTURE_BORDER_COLOR
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            border = (np.array(self.no_data_color, dtype=np.float32)).tolist()
            from OpenGL.GL import glTexParameterfv
            glTexParameterfv(GL_TEXTURE_1D, GL_TEXTURE_BORDER_COLOR, border)
        except Exception:
            try:
                glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            except Exception:
                glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, 0x2900)  # GL_CLAMP
        glBindTexture(GL_TEXTURE_1D, 0)

    def _recompute_colors(self) -> None:
        if self._scalars is None:
            return
        # Non-finite scalars map to ``NAN_TEXCOORD`` so the 1D LUT samples the
        # texture border colour (:attr:`no_data_color`) — a fixed "no data"
        # tint on every triangle, including all-NaN faces (no holes / no
        # closest-neighbour smear into the cut boundary).
        scalars_for_lut = np.asarray(self._scalars, dtype=np.float64)

        vmin, vmax = self._resolved_range_from(scalars_for_lut)
        tc = cm.scalars_to_texcoord_u(
            scalars_for_lut,
            vmin=vmin,
            vmax=vmax,
            color_mode=self.color_mode,
            piece_knots=self.piece_knots,
            n_bins=self.n_bins,
        )
        self._texcoords_flat = np.ascontiguousarray(tc).reshape(-1)

        if self._indices is not None:
            self._draw_indices = self._indices

        self._upload_lut_texture()
        self._notify_color_listeners()
        self._mesh_gl_build_or_refresh(static=False)

    def _resolved_range_from(self, scalars: np.ndarray) -> tuple[float, float]:
        """Same logic as :meth:`_resolved_range` but using a caller-supplied buffer."""
        if scalars is None or scalars.size == 0:
            return 0.0, 1.0
        if not self.auto_range and self.vmin is not None and self.vmax is not None:
            return float(self.vmin), float(self.vmax)
        finite = np.isfinite(scalars)
        if not finite.any():
            return 0.0, 1.0
        valid = scalars[finite]
        smin, smax = np.percentile(valid, [2.0, 98.0])
        return float(smin), float(smax)

    def _dispose_mesh_gl_resources(self) -> None:
        """Delete mesh VAO/VBO/EBO and optional shader program."""
        if not _GL_AVAILABLE:
            return
        try:
            from OpenGL.GL import GL_ARRAY_BUFFER, glBindBuffer, glDeleteBuffers, glDeleteProgram

            if self._mesh_vao:
                try:
                    from OpenGL.GL import glDeleteVertexArrays

                    glDeleteVertexArrays(1, np.array([self._mesh_vao], dtype=np.uint32))
                except Exception:
                    pass
                self._mesh_vao = 0
            for gid in (self._mesh_vbo_pos, self._mesh_vbo_norm, self._mesh_vbo_attr, self._mesh_ebo):
                if gid:
                    try:
                        glDeleteBuffers(1, np.array([gid], dtype=np.uint32))
                    except Exception:
                        pass
            self._mesh_vbo_pos = self._mesh_vbo_norm = self._mesh_vbo_attr = self._mesh_ebo = 0
            if self._mesh_gl_prog:
                glDeleteProgram(int(self._mesh_gl_prog))
            self._mesh_gl_prog = None
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        except Exception:
            traceback.print_exc()
        self._mesh_gl_prog_ok = None
        self._mesh_gl_loc_vert = self._mesh_gl_loc_norm = self._mesh_gl_loc_st = -1
        self._mesh_gl_u_nodata = self._mesh_gl_u_cmap = -1
        self._mesh_gl_nverts = 0

    def _dispose_mesh_gl_geometry_only(self) -> None:
        """Drop VAO/VBO/EBO but keep a compiled program (same mesh vertex count path)."""
        if not _GL_AVAILABLE:
            return
        try:
            from OpenGL.GL import GL_ARRAY_BUFFER, glBindBuffer, glDeleteBuffers

            if self._mesh_vao:
                try:
                    from OpenGL.GL import glDeleteVertexArrays

                    glDeleteVertexArrays(1, np.array([self._mesh_vao], dtype=np.uint32))
                except Exception:
                    pass
                self._mesh_vao = 0
            for gid in (self._mesh_vbo_pos, self._mesh_vbo_norm, self._mesh_vbo_attr, self._mesh_ebo):
                if gid:
                    try:
                        glDeleteBuffers(1, np.array([gid], dtype=np.uint32))
                    except Exception:
                        pass
            self._mesh_vbo_pos = self._mesh_vbo_norm = self._mesh_vbo_attr = self._mesh_ebo = 0
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        except Exception:
            traceback.print_exc()
        self._mesh_gl_nverts = 0

    def _ensure_mesh_gl_program(self) -> bool:
        """Compile VS+GS+FS once; ``False`` means fall back to fixed-function."""
        if not _GL_AVAILABLE or self._mesh_gl_prog_ok is False:
            return False
        if self._mesh_gl_prog_ok is True and self._mesh_gl_prog:
            return True
        from . import mesh_lit_program as mgp

        prog = mgp.try_compile_mesh_lit_program()
        if prog is None:
            self._mesh_gl_prog_ok = False
            return False
        lv, ln, lst = mgp.attrib_locations(prog)
        if lv < 0 or ln < 0 or lst < 0:
            try:
                from OpenGL.GL import glDeleteProgram

                glDeleteProgram(int(prog))
            except Exception:
                pass
            self._mesh_gl_prog_ok = False
            return False
        self._mesh_gl_prog = int(prog)
        self._mesh_gl_loc_vert = lv
        self._mesh_gl_loc_norm = ln
        self._mesh_gl_loc_st = lst
        self._mesh_gl_u_nodata = mgp.uniform_no_data_location(prog)
        self._mesh_gl_u_cmap = mgp.uniform_cmap_location(prog)
        self._mesh_gl_prog_ok = True
        return True

    def _mesh_gl_build_or_refresh(self, *, static: bool) -> None:
        """Upload mesh VBOs; ``static`` rebuilds geometry + element buffer."""
        if not _GL_AVAILABLE or not self._mesh_loaded:
            return
        if (
            self._verts_flat is None
            or self._normals_flat is None
            or self._indices is None
            or self._texcoords_flat is None
            or self._scalars is None
        ):
            return
        if not self._ensure_mesh_gl_program():
            return
        n = int(self._verts_flat.size // 3)
        if n <= 0:
            return
        try:
            from OpenGL.GL import (
                GL_ARRAY_BUFFER,
                GL_ELEMENT_ARRAY_BUFFER,
                GL_FALSE,
                GL_FLOAT,
                GL_STATIC_DRAW,
                glBindBuffer,
                glBindVertexArray,
                glBufferData,
                glBufferSubData,
                glEnableVertexAttribArray,
                glGenBuffers,
                glGenVertexArrays,
                glVertexAttribPointer,
            )
        except Exception:
            traceback.print_exc()
            return

        u = np.asarray(self._texcoords_flat, dtype=np.float32).reshape(-1)
        ok = np.isfinite(np.asarray(self._scalars, dtype=np.float64)).astype(np.float32).reshape(-1)
        if u.size != n or ok.size != n:
            return
        attr = np.ascontiguousarray(np.column_stack([u, ok]), dtype=np.float32)

        need_static = static or self._mesh_vbo_pos == 0 or n != self._mesh_gl_nverts
        if need_static:
            self._dispose_mesh_gl_geometry_only()
            if not self._ensure_mesh_gl_program():
                return
            vbuf = np.asarray(self._verts_flat, dtype=np.float32)
            nbuf = np.asarray(self._normals_flat, dtype=np.float32)
            ibuf = np.asarray(self._indices, dtype=np.uint32)
            abuf = np.ascontiguousarray(attr, dtype=np.float32)

            try:
                gen4 = glGenBuffers(4)
                gen4 = np.asarray(gen4, dtype=np.uint32).ravel()
                self._mesh_vbo_pos = int(gen4[0])
                self._mesh_vbo_norm = int(gen4[1])
                self._mesh_vbo_attr = int(gen4[2])
                self._mesh_ebo = int(gen4[3])
            except Exception:
                buf_ids = np.zeros(4, dtype=np.uint32)
                glGenBuffers(4, buf_ids)
                self._mesh_vbo_pos = int(buf_ids[0])
                self._mesh_vbo_norm = int(buf_ids[1])
                self._mesh_vbo_attr = int(buf_ids[2])
                self._mesh_ebo = int(buf_ids[3])

            try:
                vao_ids = glGenVertexArrays(1)
                vao_ids = np.asarray(vao_ids, dtype=np.uint32).ravel()
                self._mesh_vao = int(vao_ids[0])
            except Exception:
                vao_ids = np.zeros(1, dtype=np.uint32)
                try:
                    glGenVertexArrays(1, vao_ids)
                    self._mesh_vao = int(vao_ids[0])
                except Exception:
                    self._mesh_vao = 0

            zptr = ctypes.c_void_p(0)
            if self._mesh_vao:
                glBindVertexArray(self._mesh_vao)

            glBindBuffer(GL_ARRAY_BUFFER, self._mesh_vbo_pos)
            glBufferData(GL_ARRAY_BUFFER, vbuf.nbytes, vbuf, GL_STATIC_DRAW)
            glEnableVertexAttribArray(self._mesh_gl_loc_vert)
            glVertexAttribPointer(
                self._mesh_gl_loc_vert, 3, GL_FLOAT, GL_FALSE, 0, zptr
            )

            glBindBuffer(GL_ARRAY_BUFFER, self._mesh_vbo_norm)
            glBufferData(GL_ARRAY_BUFFER, nbuf.nbytes, nbuf, GL_STATIC_DRAW)
            glEnableVertexAttribArray(self._mesh_gl_loc_norm)
            glVertexAttribPointer(
                self._mesh_gl_loc_norm, 3, GL_FLOAT, GL_FALSE, 0, zptr
            )

            glBindBuffer(GL_ARRAY_BUFFER, self._mesh_vbo_attr)
            glBufferData(GL_ARRAY_BUFFER, abuf.nbytes, abuf, GL_STATIC_DRAW)
            glEnableVertexAttribArray(self._mesh_gl_loc_st)
            glVertexAttribPointer(
                self._mesh_gl_loc_st, 2, GL_FLOAT, GL_FALSE, 0, zptr
            )

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._mesh_ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, ibuf.nbytes, ibuf, GL_STATIC_DRAW)

            if self._mesh_vao:
                glBindVertexArray(0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self._mesh_gl_nverts = n
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self._mesh_vbo_attr)
            glBufferSubData(GL_ARRAY_BUFFER, 0, attr.nbytes, attr)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _draw_mesh_with_glsl(self) -> bool:
        """Lit mesh via VS+GS+FS (dominant-barycentric colour). Returns ``True`` if drawn."""
        if (
            not _GL_AVAILABLE
            or self._mesh_gl_prog_ok is not True
            or self._mesh_gl_prog is None
            or self._mesh_vbo_pos == 0
            or self._indices is None
            or self._tex_id is None
        ):
            return False
        try:
            from OpenGL.GL import (
                GL_TEXTURE0,
                GL_TEXTURE_1D,
                GL_TRIANGLES,
                GL_UNSIGNED_INT,
                glActiveTexture,
                glBindTexture,
                glBindVertexArray,
                glDrawElements,
                glEnable,
                glUniform1i,
                glUniform4fv,
                glUseProgram,
            )
        except Exception:
            return False

        glUseProgram(int(self._mesh_gl_prog))
        if self._mesh_gl_u_nodata >= 0:
            nd = np.array(self.no_data_color, dtype=np.float32)
            glUniform4fv(self._mesh_gl_u_nodata, 1, nd)
        if self._mesh_gl_u_cmap >= 0:
            glActiveTexture(GL_TEXTURE0)
            glUniform1i(self._mesh_gl_u_cmap, 0)
        glEnable(GL_TEXTURE_1D)
        glBindTexture(GL_TEXTURE_1D, int(self._tex_id))

        idx = self._indices
        zptr = ctypes.c_void_p(0)
        if self._mesh_vao:
            glBindVertexArray(self._mesh_vao)
            glDrawElements(GL_TRIANGLES, int(idx.size), GL_UNSIGNED_INT, zptr)
            glBindVertexArray(0)
        else:
            from OpenGL.GL import (
                GL_ARRAY_BUFFER,
                GL_ELEMENT_ARRAY_BUFFER,
                GL_FALSE,
                GL_FLOAT,
                glBindBuffer,
                glEnableVertexAttribArray,
                glVertexAttribPointer,
            )

            glBindBuffer(GL_ARRAY_BUFFER, self._mesh_vbo_pos)
            glEnableVertexAttribArray(self._mesh_gl_loc_vert)
            glVertexAttribPointer(
                self._mesh_gl_loc_vert, 3, GL_FLOAT, GL_FALSE, 0, zptr
            )
            glBindBuffer(GL_ARRAY_BUFFER, self._mesh_vbo_norm)
            glEnableVertexAttribArray(self._mesh_gl_loc_norm)
            glVertexAttribPointer(
                self._mesh_gl_loc_norm, 3, GL_FLOAT, GL_FALSE, 0, zptr
            )
            glBindBuffer(GL_ARRAY_BUFFER, self._mesh_vbo_attr)
            glEnableVertexAttribArray(self._mesh_gl_loc_st)
            glVertexAttribPointer(
                self._mesh_gl_loc_st, 2, GL_FLOAT, GL_FALSE, 0, zptr
            )
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._mesh_ebo)
            glDrawElements(GL_TRIANGLES, int(idx.size), GL_UNSIGNED_INT, zptr)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindTexture(GL_TEXTURE_1D, 0)
        glUseProgram(0)
        return True

    # ------------------------------------------------------- mesh / electrodes
    def _load_mesh(self) -> None:
        carto = self.carto
        # Reset cached interpolation artefacts; mesh geometry is changing.
        # The worker will see ``_interp_invalidate`` and clear the cache on
        # its own thread before serving the next request — this keeps cache
        # writes single-threaded.
        self._invalidate_interp_caches()
        self._dispose_mesh_gl_resources()
        self._mean_edge = 0.0

        base_field = self.scalar_field if not str(self.scalar_field).startswith("delta:") else "bipolar"
        needs_parse = (
            getattr(carto, "vertices", None) is None
            or getattr(carto, "triangles", None) is None
            or getattr(carto, base_field, None) is None
        )
        if needs_parse:
            carto.pars_mesh_file_with_electrode()

        verts = np.asarray(carto.vertices, dtype=np.float32)
        tris = np.asarray(carto.triangles, dtype=np.int32)
        if str(self.scalar_field).startswith("delta:"):
            # Will be filled in by _compute_delta_interpolated below.
            self._scalars = np.full(verts.shape[0], np.nan, dtype=np.float32)
        else:
            self._scalars = np.asarray(getattr(carto, self.scalar_field), dtype=np.float32)

        center = verts.mean(axis=0)
        radius = float(np.linalg.norm(verts - center, axis=1).max())
        if radius < 1e-9:
            radius = 1.0
        self._mesh_center = center.astype(np.float32)
        self._mesh_radius = float(radius)
        verts_norm = ((verts - center) / radius).astype(np.float32)

        norms = cm.compute_vertex_normals(verts_norm.astype(np.float64), tris.astype(np.int64))
        self._verts_flat = np.ascontiguousarray(verts_norm).reshape(-1)
        self._normals_flat = np.ascontiguousarray(norms).reshape(-1)
        self._indices = np.ascontiguousarray(tris.astype(np.uint32)).reshape(-1)
        self._draw_indices = self._indices
        # Cache raw vertex / triangle arrays for the NaN-fill pass — using
        # the normalized coords for distances would still rank neighbours
        # correctly, but raw keeps the metric meaningful for diagnostics.
        self._mesh_verts_raw = np.asarray(verts, dtype=np.float64)
        self._mesh_tris_raw = np.asarray(tris, dtype=np.int64)
        self._mesh_loaded = True
        try:
            self._mean_edge = float(
                lap.mean_edge_length(verts.astype(np.float64), tris.astype(np.int64))
            )
        except Exception:
            traceback.print_exc()
            self._mean_edge = 0.0

        # Bake per-triangle unwrapped vertex/color arrays for picking.
        n_tris = tris.shape[0]
        # ID = (n_electrodes_max) + 1..n_tris ; offset will be re-applied when
        # we set electrodes (since # of electrodes is known then). Cache the
        # unwrapped vertex positions here and rebuild colors on-demand.
        unwrapped = verts_norm[tris.reshape(-1)]
        self._pick_mesh_verts = np.ascontiguousarray(unwrapped, dtype=np.float32).reshape(-1)
        self._rebuild_mesh_pick_colors()

        # Re-normalize any electrodes that were set before the mesh loaded.
        if self._elec_xyz_raw.shape[0] > 0:
            self._normalize_electrodes()

        # If we were viewing a delta:* field, recompute now that the mesh is back.
        if str(self.scalar_field).startswith("delta:"):
            self._compute_delta_interpolated()
        self._recompute_colors()
        self._mesh_gl_build_or_refresh(static=True)

    def _rebuild_mesh_pick_colors(self) -> None:
        """Bake per-triangle RGB ID colors (3 verts × 3 bytes per triangle)."""
        if self._indices is None or self._pick_mesh_verts is None:
            return
        n_tris = self._indices.size // 3
        n_sph = self._elec_xyz_norm.shape[0]
        tri_ids = np.arange(n_tris, dtype=np.uint32) + 1 + n_sph
        rgb = pk.id_to_rgb(tri_ids)  # n_tris x 3
        per_vert = np.repeat(rgb, 3, axis=0)
        self._pick_mesh_colors = np.ascontiguousarray(per_vert, dtype=np.uint8).reshape(-1)

    def _normalize_electrodes(self) -> None:
        if self._elec_xyz_raw.shape[0] == 0:
            self._elec_xyz_norm = np.zeros((0, 3), dtype=np.float32)
            self._elec_xyz_proj_norm = np.zeros((0, 3), dtype=np.float32)
            return
        norm = (self._elec_xyz_raw - self._mesh_center) / max(self._mesh_radius, 1e-9)
        self._elec_xyz_norm = norm.astype(np.float32)
        self._compute_projected_positions()

    def _compute_projected_positions(self) -> None:
        """Project raw electrode positions onto the closest mesh surface point.

        Stored in normalized mesh coordinates so the renderer can use them
        directly. Falls back to ``_elec_xyz_norm`` when the mesh isn't loaded
        yet or the projection backend errors out.
        """
        if self._elec_xyz_raw.shape[0] == 0:
            self._elec_xyz_proj_norm = np.zeros((0, 3), dtype=np.float32)
            return
        if not self._mesh_loaded:
            self._elec_xyz_proj_norm = self._elec_xyz_norm.copy()
            return
        try:
            verts_raw = np.asarray(self.carto.vertices, dtype=np.float64)
            tris = np.asarray(self.carto.triangles, dtype=np.int64)
            projected = project_points_to_mesh(verts_raw, tris, self._elec_xyz_raw)
        except Exception:
            traceback.print_exc()
            self._elec_xyz_proj_norm = self._elec_xyz_norm.copy()
            return
        self._elec_xyz_proj_norm = (
            (projected - self._mesh_center) / max(self._mesh_radius, 1e-9)
        ).astype(np.float32)

    def _active_elec_positions(self) -> np.ndarray:
        if (
            self.use_projected
            and self._elec_xyz_proj_norm.shape[0] == self._elec_xyz_norm.shape[0]
            and self._elec_xyz_proj_norm.shape[0] > 0
        ):
            return self._elec_xyz_proj_norm
        return self._elec_xyz_norm

    def _current_sphere_radius(self) -> float:
        return float(self._SPHERE_RADIUS_FRAC * max(0.05, float(self.sphere_radius_factor)))

    # -------------------------------------------------- delta interpolation
    def set_delta_provider(self, provider) -> None:
        """Register an object exposing the delta metrics API.

        The provider only needs two methods:

        - ``get_delta_metric_keys() -> list[str]``
        - ``get_delta_values_for(key: str) -> dict[int, float]``  (mapping
          ``global_idx -> scalar`` for electrodes that already have a value).
        """
        self._delta_provider = provider
        self._notify_fields_changed()

    # ---------------------------------------------- acquisition patch preview
    def begin_patch_preview(self, field_label: str = "patch:unipolar") -> bool:
        """Take over mesh coloring for :class:`AcquisitionPatchWindow` until :meth:`end_patch_preview`."""
        if not self._mesh_loaded:
            try:
                self._load_mesh()
            except Exception:
                traceback.print_exc()
        if not self._mesh_loaded:
            return False
        if not self._patch_preview_active:
            snap_scalars = None
            if self._scalars is not None:
                snap_scalars = np.asarray(self._scalars, dtype=np.float32).copy()
            self._patch_preview_snap = {
                "scalar_field": self.scalar_field,
                "scalars": snap_scalars,
                "auto_range": self.auto_range,
                "vmin": self.vmin,
                "vmax": self.vmax,
                "current_delta_metric": self._current_delta_metric,
                "color_snap": self.snapshot_color_settings(),
            }
            self._patch_preview_active = True
        self._patch_vector_bases = None
        self._patch_vector_dirs = None
        self._patch_vector_mags = None
        self.scalar_field = str(field_label)
        self._current_delta_metric = None
        self.auto_range = True
        self.vmin = None
        self.vmax = None
        n = int(np.asarray(self.carto.vertices).shape[0])
        self._scalars = np.full(n, np.nan, dtype=np.float32)
        self._recompute_colors()
        self._request_redraw()
        return True

    def set_patch_preview_field(self, scalars: np.ndarray) -> None:
        """Update harmonic field on the main mesh (full vertex array, NaN outside patch)."""
        if not self._patch_preview_active or not self._mesh_loaded:
            return
        n = int(np.asarray(self.carto.vertices).shape[0])
        s = np.asarray(scalars, dtype=np.float32).reshape(-1)
        if s.size != n:
            raise ValueError(f"patch scalars length {s.size} != mesh vertices {n}")
        self._scalars = s
        # Keep ``auto_range`` / ``vmin`` / ``vmax`` set by
        # :meth:`set_patch_preview_color_range` — do not reset per time step.
        self._recompute_colors()
        self._request_redraw()

    def set_patch_preview_scalars(self, scalars: np.ndarray) -> None:
        """Alias for :meth:`set_patch_preview_field`."""
        self.set_patch_preview_field(scalars)

    def set_patch_preview_label(self, field_label: str) -> None:
        if self._patch_preview_active:
            self.scalar_field = str(field_label)
            self._request_redraw()

    def set_patch_preview_color_range(
        self, vmin: float, vmax: float, *, auto_range: bool = False
    ) -> None:
        """Fixed colorbar limits for patch preview (global over all time samples)."""
        if not self._patch_preview_active or not self._mesh_loaded:
            return
        self.auto_range = bool(auto_range)
        if not auto_range:
            self.vmin = float(vmin)
            self.vmax = float(vmax)
        else:
            self.vmin = None
            self.vmax = None
        if self._scalars is not None:
            self._recompute_colors()
        self._request_redraw()

    def set_patch_preview_vectors(
        self,
        bases: np.ndarray | None,
        directions: np.ndarray | None,
        magnitudes: np.ndarray | None,
        *,
        scale: float | None = None,
    ) -> None:
        """Overlay arrows at electrode neighbourhoods (patch preview only)."""
        if not self._patch_preview_active:
            return
        if bases is None or directions is None or magnitudes is None:
            self._patch_vector_bases = None
            self._patch_vector_dirs = None
            self._patch_vector_mags = None
        else:
            self._patch_vector_bases = np.asarray(bases, dtype=np.float64).reshape(-1, 3)
            self._patch_vector_dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
            self._patch_vector_mags = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
            # Vectors always use |magnitude|; direction comes from geometry only.
            self._patch_vector_mags = np.abs(self._patch_vector_mags)
            n = self._patch_vector_bases.shape[0]
            if self._patch_vector_dirs.shape[0] != n or self._patch_vector_mags.shape[0] != n:
                raise ValueError("patch vector array length mismatch")
            fin = self._patch_vector_mags[np.isfinite(self._patch_vector_mags)]
            self._patch_vector_mag_max = (
                float(np.max(np.abs(fin))) if fin.size else 1.0
            )
            if self._patch_vector_mag_max <= 0:
                self._patch_vector_mag_max = 1.0
        if scale is not None:
            self._patch_vector_scale = max(1e-9, float(scale))
        self._request_redraw()

    def set_patch_preview_anchors(
        self,
        anchor_idx: np.ndarray | None,
        labels: list[str] | None = None,
        *,
        selected: int | None = None,
    ) -> None:
        """Highlight patch electrode vertices (mesh indices) during patch preview."""
        if not self._patch_preview_active:
            return
        if anchor_idx is None:
            self._patch_anchor_idx = None
            self._patch_anchor_labels = []
            self._selected_patch_anchor = None
        else:
            self._patch_anchor_idx = np.asarray(anchor_idx, dtype=np.int64).ravel()
            self._patch_anchor_labels = list(labels or [])
            if selected is not None:
                self._selected_patch_anchor = int(selected)
        self._request_redraw()

    def set_patch_preview_selected_anchor(self, index: int | None) -> None:
        if not self._patch_preview_active:
            return
        self._selected_patch_anchor = None if index is None else int(index)
        self._request_redraw()

    def clear_patch_preview_vectors(self) -> None:
        self.set_patch_preview_vectors(None, None, None)

    def set_patch_preview_color_style(
        self,
        cmap_name: str | None = None,
        reverse: bool | None = None,
        n_bins: int | None = None,
        color_mode: str | None = None,
        piece_knots: list[float] | None = None,
        custom_bins: list[dict] | None = None,
    ) -> None:
        """Colormap / bin settings while patch preview is active."""
        if not self._patch_preview_active:
            return
        self.apply_color_settings(
            cmap_name=cmap_name,
            reverse=reverse,
            n_bins=n_bins,
            color_mode=color_mode,
            piece_knots=piece_knots,
            custom_bins=custom_bins,
        )

    def end_patch_preview(self) -> None:
        if not self._patch_preview_active:
            return
        snap = self._patch_preview_snap or {}
        self._patch_preview_active = False
        self._patch_preview_snap = None
        self._patch_vector_bases = None
        self._patch_vector_dirs = None
        self._patch_vector_mags = None
        self._patch_anchor_idx = None
        self._patch_anchor_labels = []
        self._selected_patch_anchor = None
        self._hover_patch_anchor = None
        color_snap = snap.get("color_snap")
        if color_snap:
            try:
                self.restore_color_settings(color_snap)
            except Exception:
                traceback.print_exc()
        self.scalar_field = snap.get("scalar_field", "bipolar")
        self._current_delta_metric = snap.get("current_delta_metric")
        self.auto_range = bool(snap.get("auto_range", True))
        self.vmin = snap.get("vmin")
        self.vmax = snap.get("vmax")
        if str(self.scalar_field).startswith("delta:") and self._mesh_loaded:
            self._compute_delta_interpolated()
        elif self.scalar_field in cm.SCALAR_FIELDS and self._mesh_loaded:
            try:
                self._scalars = np.asarray(
                    getattr(self.carto, self.scalar_field), dtype=np.float32
                ).reshape(-1)
            except Exception:
                traceback.print_exc()
                self._scalars = snap.get("scalars")
        else:
            self._scalars = snap.get("scalars")
        if self._scalars is not None:
            self._recompute_colors()
        self._request_redraw()

    def set_interpolation_enabled(self, flag: bool) -> None:
        new = bool(flag)
        if new == self.interpolation_enabled:
            return
        self.interpolation_enabled = new
        # If we're disabling interpolation while looking at a delta:* field,
        # fall back to "bipolar" so the viewer keeps showing something.
        if not new and str(self.scalar_field).startswith("delta:"):
            self.set_scalar_field("bipolar")
        self._notify_fields_changed()
        self._request_redraw()

    def available_fields(self) -> list[str]:
        base = list(cm.SCALAR_FIELDS)
        if self.interpolation_enabled and self._delta_provider is not None:
            try:
                keys = list(self._delta_provider.get_delta_metric_keys() or [])
            except Exception:
                traceback.print_exc()
                keys = []
            for k in keys:
                tag = f"delta:{k}"
                if tag not in base:
                    base.append(tag)
            # Conduction-velocity fields are produced separately via
            # compute_conduction_velocity(); list them so the user can
            # pick them after that button has run.
            for cv_key in sorted(self._cv_fields.keys()):
                tag = f"delta:{cv_key}"
                if tag not in base:
                    base.append(tag)
        return base

    def add_fields_listener(self, cb) -> None:
        if cb not in self._fields_listeners:
            self._fields_listeners.append(cb)

    def _notify_fields_changed(self) -> None:
        for cb in list(self._fields_listeners):
            try:
                cb(self)
            except Exception:
                traceback.print_exc()

    def export_vtk_deltas(self, folder: str) -> int:
        """Write one legacy ASCII ``.vtk`` per delta metric (viewer interpolation).

        Uses a VTK 4.1-style legacy polydata writer (no ``vtk`` Python package
        required). Requires :meth:`set_delta_provider` and a loaded mesh.
        """
        from . import vtk_delta_export as vde

        if self._delta_provider is None:
            raise RuntimeError("Delta provider is not registered on the viewer.")
        if not self._mesh_loaded:
            raise RuntimeError("Load a mesh before exporting VTK.")
        return int(
            vde.export_all_delta_metrics(
                folder,
                carto=self.carto,
                provider=self._delta_provider,
                elec_raw=self._elec_xyz_raw,
                elec_global_idx=list(self._elec_global_idx),
                interpolation_radius=self.interpolation_radius,
                default_radius_fn=self.default_interpolation_radius,
                global_pass=bool(self.use_global_patch_harmonic),
            )
        )

    def notify_delta_changed(self, global_idx=None) -> None:
        """Called by the app/mediator when delta entries have changed."""
        if not self.interpolation_enabled:
            return
        # The metric key set may have grown; let listeners refresh.
        self._notify_fields_changed()
        if str(self.scalar_field).startswith("delta:") and self._mesh_loaded:
            self._compute_delta_interpolated()
            self._request_redraw()

    def default_interpolation_radius(self) -> float:
        """Suggested default radius = 10× mean edge length of the loaded mesh."""
        return 10.0 * float(self._mean_edge) if self._mean_edge > 0 else 0.0

    def set_interpolation_radius(self, r: float | None) -> None:
        if r is None:
            new = None
        else:
            try:
                f = float(r)
            except (TypeError, ValueError):
                return
            new = f if (np.isfinite(f) and f > 0) else None
        if new == self.interpolation_radius:
            return
        self.interpolation_radius = new
        if (
            self.interpolation_enabled
            and self._mesh_loaded
            and str(self.scalar_field).startswith("delta:")
        ):
            self._compute_delta_interpolated()
            self._request_redraw()

    # -------------------------------------------------- conduction velocity
    def _apply_scalars_direct(self, f: np.ndarray) -> None:
        """Set ``self._scalars`` synchronously (Tk thread only).

        Bumps the interp serial so any in-flight worker result becomes
        stale, preventing a race where the worker overwrites this update.
        """
        with self._interp_cond:
            self._interp_serial += 1
            self._interp_applied_serial = self._interp_serial
            self._interp_pending = None
        if self._patch_preview_active:
            return
        self._scalars = np.asarray(f, dtype=np.float32)
        if self.auto_range:
            self.vmin = None
            self.vmax = None
        try:
            self._recompute_colors()
        except Exception:
            traceback.print_exc()
        self._request_redraw()

    def _interpolate_lat_field_sync(self, lat_key: str) -> np.ndarray | None:
        """Synchronously interpolate one LAT metric to every vertex.

        Returns ``None`` when nothing can be computed (no provider, no mesh
        loaded, no electrodes registered, or no finite anchors for ``lat_key``).
        """
        if not self._mesh_loaded or self._delta_provider is None:
            return None
        try:
            V = np.asarray(self.carto.vertices, dtype=np.float64)
            F = np.asarray(self.carto.triangles, dtype=np.int64)
        except Exception:
            traceback.print_exc()
            return None
        n = int(V.shape[0])
        if n == 0:
            return None
        try:
            raw_vals = self._delta_provider.get_delta_values_for(lat_key) or {}
        except Exception:
            traceback.print_exc()
            return None
        if not raw_vals:
            return None
        if self._L is None or int(self._L.shape[0]) != n:
            try:
                self._L = lap.cot_laplacian(V, F)
            except Exception:
                traceback.print_exc()
                self._L = None
                return None
        if self._anchor_vidx is None and self._elec_xyz_raw.size > 0:
            try:
                self._anchor_vidx = lap.map_points_to_vertices(V, self._elec_xyz_raw)
            except Exception:
                traceback.print_exc()
                self._anchor_vidx = None
        if self._anchor_vidx is None or self._anchor_vidx.size == 0:
            return None

        active_local: list[int] = []
        anc_val: list[float] = []
        for k, gidx in enumerate(self._elec_global_idx):
            if k >= self._anchor_vidx.size:
                break
            v = raw_vals.get(int(gidx))
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(vf):
                continue
            active_local.append(k)
            anc_val.append(vf)
        if not active_local:
            return None
        anc_idx = self._anchor_vidx[np.asarray(active_local, dtype=np.int64)]
        anc_val_arr = np.asarray(anc_val, dtype=np.float64)
        try:
            f = lap.harmonic_interpolate(self._L, anc_idx, anc_val_arr)
        except Exception:
            traceback.print_exc()
            return None
        return np.asarray(f, dtype=np.float64)

    # Mapping from LAT metric key (produced by the mediator's _LAT_DERIVATIONS)
    # to the CV metric key we expose for it. Keep S1/S2/S3 + SR.
    _CV_FROM_LAT = (
        ("lat_stim[1]", "cv_stim[1]"),
        ("lat_stim[2]", "cv_stim[2]"),
        ("lat_stim[3]", "cv_stim[3]"),
        ("lat_sinus[1]", "cv_sinus[1]"),
    )

    def compute_conduction_velocity(self) -> dict[str, int]:
        """Compute CV magnitude maps for the S1/S2/S3 stims and SR LAT.

        For each LAT field with at least one finite electrode anchor, we
        interpolate it across the whole mesh (cotangent harmonic), take
        the mesh gradient, and store ``1 / |grad LAT|`` (mm/ms) in
        ``self._cv_fields``. Returns ``{cv_key: n_anchors_used}`` so the
        caller can report which CV maps actually got produced.
        """
        from . import cv as cv_mod

        if not self._mesh_loaded or self._delta_provider is None:
            return {}
        try:
            V = np.asarray(self.carto.vertices, dtype=np.float64)
            F = np.asarray(self.carto.triangles, dtype=np.int64)
        except Exception:
            traceback.print_exc()
            return {}
        produced: dict[str, int] = {}
        try:
            available_keys = set(self._delta_provider.get_delta_metric_keys() or [])
        except Exception:
            traceback.print_exc()
            available_keys = set()
        for lat_key, cv_key in self._CV_FROM_LAT:
            if lat_key not in available_keys:
                self._cv_fields.pop(cv_key, None)
                continue
            lat_full = self._interpolate_lat_field_sync(lat_key)
            if lat_full is None:
                self._cv_fields.pop(cv_key, None)
                continue
            try:
                cv = cv_mod.conduction_velocity_from_lat(V, F, lat_full)
            except Exception:
                traceback.print_exc()
                self._cv_fields.pop(cv_key, None)
                continue
            self._cv_fields[cv_key] = np.asarray(cv, dtype=np.float32)
            try:
                anchors = self._delta_provider.get_delta_values_for(lat_key) or {}
            except Exception:
                anchors = {}
            produced[cv_key] = len(anchors)
        self._notify_fields_changed()
        # If a CV field is currently selected, refresh the display.
        if (
            str(self.scalar_field).startswith("delta:cv_")
            and self._current_delta_metric in self._cv_fields
        ):
            self._apply_scalars_direct(self._cv_fields[self._current_delta_metric])
        return produced

    def _compute_delta_interpolated(self) -> None:
        """Snapshot the inputs and hand off to the background worker.

        The Tk thread does only the cheap work here: read the metric values,
        decide global/radius mode, copy a few arrays. The actual Laplacian /
        Dijkstra / sparse solve runs in :meth:`_serve_interp_request` on the
        dedicated worker thread, so this method returns almost instantly even
        on large meshes. Selecting another point while a previous request is
        still in flight simply replaces the pending slot.
        """
        metric = self._current_delta_metric
        if metric is None or self._delta_provider is None:
            return
        # CV fields are precomputed by compute_conduction_velocity() and live
        # in ``_cv_fields``; no anchor solve is needed for them. If the user
        # hasn't run "Compute CV" yet, show NaN so the colormap renders the
        # "no data" tint instead of a stale field.
        if str(metric).startswith("cv_"):
            try:
                n = int(np.asarray(self.carto.vertices).shape[0])
            except Exception:
                return
            cached = self._cv_fields.get(metric)
            if cached is None or int(cached.size) != n:
                cached = np.full(n, np.nan, dtype=np.float32)
            self._apply_scalars_direct(cached)
            return
        try:
            n = int(np.asarray(self.carto.vertices).shape[0])
        except Exception:
            return

        try:
            raw_vals = self._delta_provider.get_delta_values_for(metric) or {}
            # Defensive copy: ``app.delta`` is mutated on the Tk thread while
            # this snapshot is handed to the worker — never share live dict /
            # mutable entries with the background thread.
            values: dict[int, float] = {}
            for k, v in raw_vals.items():
                try:
                    values[int(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        except Exception:
            traceback.print_exc()
            values = {}

        # Decide mode for this pass on the Tk thread (cheap). Global no longer
        # disables the geodesic cut — it now interpolates harmonically over
        # the same union-of-balls patch the local mode uses, just with one
        # harmonic system across the whole patch and natural Neumann BC at
        # the patch rim (so a single anchor fills its patch uniformly).
        global_pass = bool(self.use_global_patch_harmonic)
        rad = self.interpolation_radius
        if rad is None or not np.isfinite(float(rad)) or float(rad) <= 0:
            d = self.default_interpolation_radius()
            radius_eff: float | None = float(d) if d > 0 else None
        else:
            radius_eff = float(rad)

        req = {
            "metric": metric,
            "values": values,
            "radius_eff": radius_eff,
            "global_pass": global_pass,
            "n_verts": n,
            # Geometry + electrode snapshots so the worker never reads
            # mutating state on its thread.
            "verts_raw": np.asarray(self.carto.vertices, dtype=np.float64),
            "tris": np.asarray(self.carto.triangles, dtype=np.int64),
            "elec_raw": self._elec_xyz_raw.copy() if self._elec_xyz_raw.size else self._elec_xyz_raw,
            "elec_global_idx": list(self._elec_global_idx),
        }
        self._post_interp_request(req)

    # ------------------------------------------------------- worker plumbing
    def _start_interp_worker(self) -> None:
        if self._interp_thread is not None and self._interp_thread.is_alive():
            return
        self._interp_stop = False
        self._interp_thread = threading.Thread(
            target=self._interp_worker_loop, daemon=True, name="mesh-delta-interp"
        )
        self._interp_thread.start()

    def _post_interp_request(self, req: dict) -> None:
        """Replace the pending request slot with ``req`` and wake the worker.

        Older un-served requests are silently dropped — exactly what the user
        asked for: only the most recent click survives in the queue.
        """
        self._start_interp_worker()
        with self._interp_cond:
            self._interp_serial += 1
            req["serial"] = self._interp_serial
            self._interp_pending = req
            self._interp_cond.notify()

    def _invalidate_interp_caches(self) -> None:
        """Tell the worker to drop its caches before serving the next request."""
        with self._interp_cond:
            self._interp_invalidate = True
            # Bump serial so any in-flight result becomes stale.
            self._interp_serial += 1
            # Drop any queued snapshot: its ``serial`` no longer matches the
            # bumped ``_interp_serial``, so the worker would otherwise skip it
            # and never run a fresh ``_compute_delta_interpolated`` until some
            # unrelated UI event re-posted (mesh stayed on stale colours).
            self._interp_pending = None
            self._interp_cond.notify()

    def _interp_worker_loop(self) -> None:
        while True:
            with self._interp_cond:
                while (
                    not self._interp_stop
                    and self._interp_pending is None
                    and not self._interp_invalidate
                ):
                    self._interp_cond.wait()
                if self._interp_stop:
                    return
                if self._interp_invalidate:
                    self._interp_invalidate = False
                    # Clear caches on the worker thread (single writer rule).
                    self._L = None
                    self._mesh_graph = None
                    self._anchor_vidx = None
                    self._anchor_dist = None
                    self._anchor_dist_radius = None
                    self._anchor_dist_anchors = None
                req = self._interp_pending
                self._interp_pending = None
                latest_serial = self._interp_serial
            if req is None:
                continue
            if req["serial"] != latest_serial:
                continue
            try:
                self._serve_interp_request(req)
            except Exception:
                traceback.print_exc()

    def _serve_interp_request(self, req: dict) -> None:
        """Run on the worker thread. All heavy compute lives here."""
        serial = req["serial"]
        n = int(req["n_verts"])
        values = req["values"]
        verts_raw = req["verts_raw"]
        tris = req["tris"]
        elec_raw = req["elec_raw"]
        elec_global_idx = req["elec_global_idx"]
        radius_eff: float | None = req["radius_eff"]
        global_pass: bool = bool(req["global_pass"])

        def stale() -> bool:
            return serial != self._interp_serial

        def empty_result() -> None:
            self._post_apply(serial, np.full(n, np.nan, dtype=np.float32))

        if not values:
            empty_result()
            return

        # 1) Laplacian (cached). Built on this thread on first use.
        if self._L is None or int(self._L.shape[0]) != n:
            try:
                self._L = lap.cot_laplacian(verts_raw, tris)
            except Exception:
                traceback.print_exc()
                self._L = None
                empty_result()
                return
        if stale():
            return

        # 2) Anchor vertex index (electrode -> mesh vertex).
        if self._anchor_vidx is None and elec_raw is not None and getattr(elec_raw, "size", 0) > 0:
            try:
                self._anchor_vidx = lap.map_points_to_vertices(verts_raw, elec_raw)
            except Exception:
                traceback.print_exc()
                self._anchor_vidx = None
        if stale():
            return

        anc_idx_all = self._anchor_vidx
        if anc_idx_all is None or anc_idx_all.size == 0:
            empty_result()
            return

        # 3) Which anchors have a finite value this metric.
        active_local: list[int] = []
        anc_val_list: list[float] = []
        for k, gidx in enumerate(elec_global_idx):
            if k >= anc_idx_all.size:
                break
            v = values.get(gidx)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(vf):
                continue
            active_local.append(k)
            anc_val_list.append(vf)
        if not active_local:
            empty_result()
            return
        active_local_arr = np.asarray(active_local, dtype=np.int64)
        anc_idx = anc_idx_all[active_local_arr]
        anc_val = np.asarray(anc_val_list, dtype=np.float64)
        if stale():
            return

        # 4) Geodesic distance cache (we always need a radius now — even
        # "global" mode is bounded on the union-of-balls patch).
        #
        # **Edge Dijkstra first** (SciPy ``csgraph`` — fast C code). Triangle
        # FMM is more accurate but our pure-Python implementation is far slower
        # on large meshes when run uncapped for every anchor. Heat method is
        # last resort if the graph is unavailable.
        if radius_eff is not None:
            if self._mesh_graph is None:
                try:
                    self._mesh_graph = lap.build_mesh_graph(verts_raw, tris)
                except Exception:
                    traceback.print_exc()
                    self._mesh_graph = None
            anc_tuple = tuple(int(x) for x in anc_idx_all.tolist())
            r_eff = float(radius_eff)
            tol = 1e-5 * max(1.0, r_eff)
            uncapped_cached = (
                self._anchor_dist is not None
                and self._anchor_dist_anchors == anc_tuple
                and self._anchor_dist_radius is not None
                and math.isinf(float(self._anchor_dist_radius))
            )
            capped_cached = (
                self._anchor_dist is not None
                and self._anchor_dist_anchors == anc_tuple
                and self._anchor_dist_radius is not None
                and math.isfinite(float(self._anchor_dist_radius))
                and abs(float(self._anchor_dist_radius) - r_eff) <= tol
            )
            cache_fresh = uncapped_cached or capped_cached
            if not cache_fresh:
                if stale():
                    return
                dist = None
                # 1) Edge Dijkstra uncapped (preferred — fast).
                if self._mesh_graph is not None:
                    try:
                        dist = lap.dijkstra_distances_from_anchors(
                            self._mesh_graph, anc_idx_all, None
                        )
                    except Exception:
                        traceback.print_exc()
                        dist = None
                if dist is None:
                    try:
                        dist = lap.fmm_distances_from_anchors(
                            verts_raw, tris, anc_idx_all, None
                        )
                    except Exception:
                        traceback.print_exc()
                        dist = None
                if dist is None:
                    try:
                        dist = lap.heat_geodesic_distances_from_anchors(
                            verts_raw,
                            tris,
                            anc_idx_all,
                            radius=None,
                            tau_reference_radius=r_eff,
                        )
                    except Exception:
                        traceback.print_exc()
                        dist = None
                if dist is not None:
                    self._anchor_dist = dist
                    self._anchor_dist_anchors = anc_tuple
                    self._anchor_dist_radius = float("inf")
                else:
                    self._anchor_dist = None
                    self._anchor_dist_anchors = None
                    self._anchor_dist_radius = None
        if stale():
            return

        # 5) Solve.
        solve_mode = "global" if global_pass else "local"
        try:
            if radius_eff is None:
                # No mesh geometry to derive a radius from — fall back to the
                # unbounded harmonic so the user at least sees something.
                f = lap.harmonic_interpolate(self._L, anc_idx, anc_val)
            elif self._anchor_dist is not None:
                mask = np.zeros(self._anchor_dist.shape[0], dtype=bool)
                mask[active_local_arr] = True
                f = lap.harmonic_interpolate_bounded_cached(
                    self._L,
                    self._anchor_dist,
                    anc_idx,
                    anc_val,
                    mask,
                    float(radius_eff),
                    mode=solve_mode,
                )
            elif self._mesh_graph is not None:
                f = lap.harmonic_interpolate_bounded(
                    self._L,
                    self._mesh_graph,
                    anc_idx,
                    anc_val,
                    float(radius_eff),
                    mode=solve_mode,
                )
            else:
                f = lap.harmonic_interpolate(self._L, anc_idx, anc_val)
        except Exception:
            traceback.print_exc()
            f = np.full(n, np.nan, dtype=np.float64)

        self._post_apply(serial, np.asarray(f, dtype=np.float32))

    def _post_apply(self, serial: int, f: np.ndarray) -> None:
        """Schedule the result onto the Tk thread (only OpenGL-safe place)."""
        def apply() -> None:
            # Reject if the user has since posted a newer request OR if an
            # even newer result was already applied.
            if serial < self._interp_applied_serial:
                return
            if serial != self._interp_serial:
                return
            self._interp_applied_serial = serial
            if self._patch_preview_active:
                return
            self._scalars = f
            if self.auto_range:
                self.vmin = None
                self.vmax = None
            self._recompute_colors()
            self._request_redraw()

        try:
            self.after(0, apply)
        except Exception:
            traceback.print_exc()

    def set_electrodes(
        self,
        positions_xyz: np.ndarray,
        global_indices: list[int] | np.ndarray,
        labels: list[str] | None = None,
    ) -> None:
        """Register electrode positions (mesh-frame, raw units).

        Parameters
        ----------
        positions_xyz : (N, 3) array-like (raw, same units as carto mesh)
        global_indices : list of N ints (App-side flat row index per electrode)
        labels : optional list of N display strings (e.g. ``"P12"``)
        """
        pts = np.asarray(positions_xyz, dtype=np.float64).reshape(-1, 3)
        gidx = [int(x) for x in (global_indices or [])]
        if pts.shape[0] != len(gidx):
            raise ValueError("positions/global_indices length mismatch")
        labels = list(labels or [f"P{i}" for i in gidx])
        if len(labels) != len(gidx):
            raise ValueError("labels length mismatch")

        self._elec_xyz_raw = pts
        self._elec_global_idx = gidx
        self._elec_labels = labels
        # Anchor-vertex cache is keyed off these positions; invalidate it.
        self._invalidate_interp_caches()
        self._normalize_electrodes()
        self._rebuild_mesh_pick_colors()  # tri ids depend on n_electrodes
        # If we're already showing a delta:* map, re-interpolate with the new anchors.
        if self._mesh_loaded and str(self.scalar_field).startswith("delta:"):
            self._compute_delta_interpolated()
        self._request_redraw()

    def set_selected_global_index(self, idx: int | None) -> None:
        if idx is None:
            self._selected_global_idx = None
        else:
            self._selected_global_idx = int(idx)
        self._request_redraw()

    def get_hover_info(self) -> dict:
        return {
            "kind": self._hover_kind,
            "global_idx": self._hover_global_idx,
            "triangle": self._hover_triangle,
            "mesh_name": self.mesh_name,
            "cursor": self._cursor_pos,
        }

    # ----------------------------------------------------------- color bar GL
    def _colorbar_rect(self, w: int, h: int) -> tuple[float, float, float, float] | None:
        if not self.show_colorbar:
            return None
        t = self._CB_THICK
        if self.cb_orientation == "vertical":
            L = min(220, max(80, int(h * 0.5)))
            x1 = float(w - self._CB_PAD + self.cb_drag_dx)
            x0 = x1 - t
            cy = h / 2.0 + self.cb_drag_dy
            y0 = cy - L / 2.0
            y1 = cy + L / 2.0
        else:
            L = min(280, max(100, int(w * 0.55)))
            y1 = float(h - self._CB_PAD + self.cb_drag_dy)
            y0 = y1 - t
            cx = w / 2.0 + self.cb_drag_dx
            x0 = cx - L / 2.0
            x1 = cx + L / 2.0
        if x1 < 2 or y1 < 2 or x0 > w - 2 or y0 > h - 2:
            return max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
        return x0, y0, x1, y1

    def _hit_colorbar(self, x: int, y: int) -> bool:
        r = self._cb_rect
        if r is None:
            return False
        x0, y0, x1, y1 = r
        return x0 <= x <= x1 and y0 <= y <= y1

    @staticmethod
    def _fmt_value(v: float) -> str:
        if not np.isfinite(v):
            return "n/a"
        a = abs(v)
        if a == 0.0:
            return "0"
        if a >= 1e4 or a < 1e-2:
            return f"{v:.2e}"
        return f"{v:.3g}"

    def _draw_colorbar_gl(self, w: int, h: int) -> None:
        r = self._colorbar_rect(w, h)
        self._cb_rect = r
        if r is None:
            return
        x0, y0, x1, y1 = r
        vmin, vmax = self._resolved_range()
        if vmax - vmin < 1e-12:
            vmax = vmin + 1.0

        nseg = 96
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_1D)
        glDisable(GL_TEXTURE_2D)
        self._enter_ortho(w, h)

        if self.cb_orientation == "vertical":
            vals = np.linspace(vmax, vmin, nseg)
            ys = np.linspace(y0, y1, nseg)
            rgb = self._scalar_values_to_rgb(vals)
            for i in range(nseg - 1):
                r0, g0, b0 = rgb[i]
                r1, g1, b1 = rgb[i + 1]
                ya, yb = ys[i], ys[i + 1]
                glBegin(GL_QUADS)
                glColor3f(float(r0), float(g0), float(b0))
                glVertex2f(x0, ya)
                glVertex2f(x1, ya)
                glColor3f(float(r1), float(g1), float(b1))
                glVertex2f(x1, yb)
                glVertex2f(x0, yb)
                glEnd()
        else:
            vals = np.linspace(vmin, vmax, nseg)
            xs = np.linspace(x0, x1, nseg)
            rgb = self._scalar_values_to_rgb(vals)
            for i in range(nseg - 1):
                r0, g0, b0 = rgb[i]
                r1, g1, b1 = rgb[i + 1]
                xa, xb = xs[i], xs[i + 1]
                glBegin(GL_QUADS)
                glColor3f(float(r0), float(g0), float(b0))
                glVertex2f(xa, y0)
                glVertex2f(xb, y0)
                glColor3f(float(r1), float(g1), float(b1))
                glVertex2f(xb, y1)
                glVertex2f(xa, y1)
                glEnd()

        # Outline.
        glColor3f(0.85, 0.85, 0.9)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glVertex2f(x0, y0)
        glVertex2f(x1, y0)
        glVertex2f(x1, y0)
        glVertex2f(x1, y1)
        glVertex2f(x1, y1)
        glVertex2f(x0, y1)
        glVertex2f(x0, y1)
        glVertex2f(x0, y0)
        glEnd()

        # Ticks + numeric labels.
        n_ticks = 5
        glColor3f(0.85, 0.85, 0.9)
        for k in range(n_ticks):
            frac = k / float(n_ticks - 1)
            if self.cb_orientation == "vertical":
                val = vmax - frac * (vmax - vmin)
                yy = y0 + frac * (y1 - y0)
                glBegin(GL_LINES)
                glVertex2f(x0, yy)
                glVertex2f(x0 - 5, yy)
                glEnd()
            else:
                val = vmin + frac * (vmax - vmin)
                xx = x0 + frac * (x1 - x0)
                glBegin(GL_LINES)
                glVertex2f(xx, y0)
                glVertex2f(xx, y0 - 5)
                glEnd()

        # Numeric labels via PIL-rasterized text textures.
        if self._text is None:
            self._text = TextRenderer(font_size=13)
        for k in range(n_ticks):
            frac = k / float(n_ticks - 1)
            if self.cb_orientation == "vertical":
                val = vmax - frac * (vmax - vmin)
                yy = y0 + frac * (y1 - y0)
                self._text.draw(self._fmt_value(float(val)), x0 - 8, yy, anchor="e")
            else:
                val = vmin + frac * (vmax - vmin)
                xx = x0 + frac * (x1 - x0)
                self._text.draw(self._fmt_value(float(val)), xx, y0 - 7, anchor="s")

        # Field title.
        if self.cb_orientation == "vertical":
            self._text.draw(self.scalar_field, (x0 + x1) / 2.0, y0 - 6, anchor="s", px=14)
        else:
            self._text.draw(self.scalar_field, x0 - 8, (y0 + y1) / 2.0, anchor="e", px=14)

        self._leave_ortho()
        glEnable(GL_DEPTH_TEST)

    # ------------------------------------------------------- camera and HUD
    def _setup_camera(self, w: int, h: int) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / float(h), 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, 0.0, self._zoom, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        glTranslatef(self._pan[0], self._pan[1], 0.0)
        glRotatef(self._rot_x, 1.0, 0.0, 0.0)
        glRotatef(self._rot_y, 0.0, 1.0, 0.0)

    def _enter_ortho(self, w: int, h: int) -> None:
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, w, h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

    def _leave_ortho(self) -> None:
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # ---------------------------------------------------- mesh rendering
    def _render_mesh_lit_fixed_function(self) -> None:
        """Classic GL pipeline: linear interpolation of 1D texture coordinates."""
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (0.2, 0.2, 0.22, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (0.88, 0.88, 0.9, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.4, 0.4, 0.42, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 48.0)

        glEnable(GL_TEXTURE_1D)
        glBindTexture(GL_TEXTURE_1D, self._tex_id)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self._verts_flat)
        glNormalPointer(GL_FLOAT, 0, self._normals_flat)
        glTexCoordPointer(1, GL_FLOAT, 0, self._texcoords_flat)
        glColor3f(1.0, 1.0, 1.0)
        idx = self._draw_indices if self._draw_indices is not None else self._indices
        glDrawElements(GL_TRIANGLES, int(idx.size), GL_UNSIGNED_INT, idx)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        glDisable(GL_TEXTURE_1D)
        glBindTexture(GL_TEXTURE_1D, 0)
        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)

    def _render_mesh_lit(self) -> None:
        if (
            self._verts_flat is None
            or self._indices is None
            or self._normals_flat is None
            or self._texcoords_flat is None
            or self._tex_id is None
        ):
            return

        glPushMatrix()
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, (0.35, 0.45, 0.95, 0.0))
        glPopMatrix()

        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.18, 0.18, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.92, 0.92, 0.88, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.45, 0.45, 0.45, 1.0))

        if self.prefer_legacy_mesh_rendering:
            self._render_mesh_lit_fixed_function()
            return

        if self._draw_mesh_with_glsl():
            return

        self._render_mesh_lit_fixed_function()

    def _render_spheres_color(self) -> None:
        positions = self._active_elec_positions()
        if positions.shape[0] == 0:
            return
        radius = self._current_sphere_radius()
        base_v = _SPH_V.astype(np.float32) * radius
        base_v_flat = np.ascontiguousarray(base_v).reshape(-1)
        base_t = _SPH_T.astype(np.uint32).reshape(-1)
        base_n = _SPH_V.astype(np.float32)
        base_n_flat = np.ascontiguousarray(base_n).reshape(-1)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.6, 0.6, 0.6, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 64.0)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, base_v_flat)
        glNormalPointer(GL_FLOAT, 0, base_n_flat)

        sel = self._selected_global_idx
        hov = self._hover_global_idx
        for k, (pos, gidx) in enumerate(zip(positions, self._elec_global_idx)):
            glPushMatrix()
            glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
            is_sel = sel is not None and int(gidx) == int(sel)
            is_hov = hov is not None and int(gidx) == int(hov)
            if is_sel:
                glColor3f(1.0, 0.92, 0.15)
            elif is_hov:
                glColor3f(0.35, 0.95, 1.0)
            else:
                glColor3f(0.92, 0.78, 0.30)
            glDrawElements(GL_TRIANGLES, int(base_t.size), GL_UNSIGNED_INT, base_t)
            glPopMatrix()

        # Render a slightly bigger highlight halo on top for the selected one
        # so it remains visible even when half-buried in the mesh.
        if sel is not None:
            for k, gidx in enumerate(self._elec_global_idx):
                if int(gidx) != int(sel):
                    continue
                pos = positions[k]
                glDisable(GL_DEPTH_TEST)
                halo_v = (_SPH_V.astype(np.float32) * radius * self._HIGHLIGHT_SCALE)
                halo_flat = np.ascontiguousarray(halo_v).reshape(-1)
                glVertexPointer(3, GL_FLOAT, 0, halo_flat)
                glPushMatrix()
                glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
                glColor4f(1.0, 0.95, 0.20, 0.45)
                glDrawElements(GL_TRIANGLES, int(base_t.size), GL_UNSIGNED_INT, base_t)
                glPopMatrix()
                glEnable(GL_DEPTH_TEST)
                glVertexPointer(3, GL_FLOAT, 0, base_v_flat)
                break

        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)

    def _render_color_scene(self, w: int, h: int) -> None:
        glClearColor(*self._BG_RGB)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Pick pass disables blending; restore it so HUD alpha works correctly.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        self._setup_camera(w, h)
        self._render_mesh_lit()
        self._render_patch_anchors()
        self._render_patch_vectors_3d()
        self._render_spheres_color()

    def _patch_anchor_positions_norm(self) -> np.ndarray:
        if self._patch_anchor_idx is None or self._verts_flat is None:
            return np.zeros((0, 3), dtype=np.float64)
        idx = np.asarray(self._patch_anchor_idx, dtype=np.int64).ravel()
        verts = np.asarray(self._verts_flat, dtype=np.float64).reshape(-1, 3)
        idx = idx[(idx >= 0) & (idx < verts.shape[0])]
        if idx.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return verts[idx]

    def _pick_pool_sizes(self) -> tuple[int, int, int]:
        n_sph = int(self._elec_xyz_norm.shape[0])
        n_tris = int(self._indices.size // 3) if self._indices is not None else 0
        n_patch = int(self._patch_anchor_idx.size) if (
            self._patch_preview_active and self._patch_anchor_idx is not None
        ) else 0
        return n_sph, n_tris, n_patch

    def _decode_pick_id(self, pick_id: int) -> tuple[str, int]:
        n_sph, n_tris, n_patch = self._pick_pool_sizes()
        if pick_id <= 0:
            return "empty", -1
        if pick_id <= n_sph:
            return "sphere", pick_id - 1
        if pick_id <= n_sph + n_tris:
            return "triangle", pick_id - 1 - n_sph
        if self._patch_preview_active and pick_id <= n_sph + n_tris + n_patch:
            return "patch_anchor", pick_id - 1 - n_sph - n_tris
        return "empty", -1

    def _render_patch_vectors_3d(self) -> None:
        """White 3D arrows on the mesh (normalized coords, always visible)."""
        if not self._patch_preview_active:
            return
        bases = self._patch_vector_bases
        dirs = self._patch_vector_dirs
        mags = self._patch_vector_mags
        if bases is None or dirs is None or mags is None:
            return

        try:
            from OpenGL.GL import glUseProgram

            glUseProgram(0)
        except Exception:
            pass

        scale = float(self._patch_vector_scale)
        if scale <= 0:
            return
        r = max(float(self._mesh_radius), 1e-9)
        norm_len_scale = scale / r

        verts: list[list[float]] = []
        bases = np.asarray(bases, dtype=np.float64).reshape(-1, 3)
        dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        mags = np.abs(np.asarray(mags, dtype=np.float64).reshape(-1))
        for i in range(bases.shape[0]):
            mag = float(mags[i])
            if not np.isfinite(mag) or mag <= 0.0:
                continue
            d = dirs[i]
            dn = float(np.linalg.norm(d))
            if dn < 1e-12 or not np.all(np.isfinite(bases[i])):
                continue
            d = d / dn
            b_n = self._raw_to_render_norm(bases[i].reshape(1, 3))[0]
            length_n = mag * norm_len_scale
            if length_n < 1e-6:
                continue
            tip_n = b_n + d * length_n
            verts.append([float(b_n[0]), float(b_n[1]), float(b_n[2])])
            verts.append([float(tip_n[0]), float(tip_n[1]), float(tip_n[2])])

        if not verts:
            return

        arr = np.ascontiguousarray(np.asarray(verts, dtype=np.float32))
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_1D)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(4.5)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, arr)
        glDrawArrays(GL_LINES, 0, int(arr.shape[0]))
        glDisableClientState(GL_VERTEX_ARRAY)
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
        glColor4f(1.0, 1.0, 1.0, 1.0)

    def _render_patch_anchors(self) -> None:
        """Small point markers on patch electrode vertices (selected = highlight)."""
        if (
            not self._patch_preview_active
            or self._patch_anchor_idx is None
            or self._verts_flat is None
            or self._patch_anchor_idx.size == 0
        ):
            return
        positions = self._patch_anchor_positions_norm()
        if positions.shape[0] == 0:
            return
        sel = self._selected_patch_anchor

        try:
            from OpenGL.GL import glUseProgram

            glUseProgram(0)
        except Exception:
            pass

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_1D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)

        pts = np.ascontiguousarray(positions, dtype=np.float32)
        glEnableClientState(GL_VERTEX_ARRAY)

        if sel is not None and 0 <= int(sel) < pts.shape[0]:
            one = np.ascontiguousarray(pts[int(sel) : int(sel) + 1, :], dtype=np.float32)
            glVertexPointer(3, GL_FLOAT, 0, one)
            glPointSize(10.0)
            glColor4f(1.0, 0.82, 0.08, 1.0)
            glDrawArrays(GL_POINTS, 0, 1)

        glDisableClientState(GL_VERTEX_ARRAY)
        glPointSize(1.0)

    @staticmethod
    def _gl_matrix4() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mv = np.array(glGetDoublev(GL_MODELVIEW_MATRIX), dtype=np.float64).reshape(4, 4, order="F")
        pr = np.array(glGetDoublev(GL_PROJECTION_MATRIX), dtype=np.float64).reshape(4, 4, order="F")
        vp = np.array(glGetDoublev(GL_VIEWPORT), dtype=np.float64).ravel()
        return mv, pr, vp

    def _project_norm_to_window(
        self, x: float, y: float, z: float, w: int, h: int
    ) -> tuple[float, float] | None:
        """Project normalized mesh coords to window pixels (top-left origin)."""
        try:
            mv, pr, vp = self._gl_matrix4()
            p = np.array([float(x), float(y), float(z), 1.0], dtype=np.float64)
            clip = pr @ (mv @ p)
            if not np.isfinite(clip[3]) or abs(float(clip[3])) < 1e-12:
                return None
            ndc = clip[:3] / clip[3]
            sx = float(vp[0] + (ndc[0] + 1.0) * 0.5 * vp[2])
            sy = float(vp[1] + (ndc[1] + 1.0) * 0.5 * vp[3])
            if not (np.isfinite(sx) and np.isfinite(sy)):
                return None
            return sx, float(h) - sy
        except Exception:
            return None

    @staticmethod
    def _draw_screen_arrow_2d(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        head_len: float = 9.0,
        head_half: float = 5.0,
    ) -> None:
        """Flat 2D arrow in pixel space (always on top of the mesh)."""
        dx = x1 - x0
        dy = y1 - y0
        seg_len = float(math.hypot(dx, dy))
        if seg_len < 0.5:
            glBegin(GL_LINES)
            glVertex2f(x0, y0)
            glVertex2f(x0 + 1.0, y0)
            glEnd()
            return
        ux, uy = dx / seg_len, dy / seg_len
        # Shorten shaft so arrowhead sits at the tip.
        shaft_end_x = x1 - ux * head_len
        shaft_end_y = y1 - uy * head_len
        px, py = -uy, ux
        glBegin(GL_LINES)
        glVertex2f(x0, y0)
        glVertex2f(shaft_end_x, shaft_end_y)
        glEnd()
        glBegin(GL_TRIANGLES)
        glVertex2f(x1, y1)
        glVertex2f(shaft_end_x + px * head_half, shaft_end_y + py * head_half)
        glVertex2f(shaft_end_x - px * head_half, shaft_end_y - py * head_half)
        glEnd()

    def _raw_to_render_norm(self, pts: np.ndarray) -> np.ndarray:
        """Carto raw mesh coordinates → same normalized frame as ``_verts_flat``."""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if not self._mesh_loaded or p.size == 0:
            return p
        c = np.asarray(self._mesh_center, dtype=np.float64).reshape(1, 3)
        r = max(float(self._mesh_radius), 1e-9)
        return (p - c) / r

    def _draw_patch_vector_overlay(self, w: int, h: int) -> None:
        """Screen-space arrows drawn last (always on top of mesh + HUD)."""
        if not self._patch_preview_active:
            return
        bases = self._patch_vector_bases
        dirs = self._patch_vector_dirs
        mags = self._patch_vector_mags
        if bases is None or dirs is None or mags is None:
            return

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        self._setup_camera(w, h)

        mag_max = max(float(self._patch_vector_mag_max), 1e-12)
        segments: list[tuple[float, float, float, float]] = []
        min_px = 22.0
        max_px = 0.42 * float(min(w, h))
        bases = np.asarray(bases, dtype=np.float64).reshape(-1, 3)
        dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        mags = np.asarray(mags, dtype=np.float64).reshape(-1)

        for i in range(bases.shape[0]):
            mag = float(mags[i])
            if not np.isfinite(mag):
                continue
            b_raw = bases[i]
            d = dirs[i]
            dn = float(np.linalg.norm(d))
            if dn < 1e-12 or not np.all(np.isfinite(b_raw)) or not np.all(np.isfinite(d)):
                continue
            d = d / dn
            mag_abs = abs(mag)

            b_n = self._raw_to_render_norm(b_raw.reshape(1, 3))[0]
            # Screen direction from a short step along the 3D direction (normalized frame).
            step = 0.04 * max(float(self._patch_vector_scale), 0.01)
            tip_n = self._raw_to_render_norm((b_raw + d * step).reshape(1, 3))[0]
            p0 = self._project_norm_to_window(float(b_n[0]), float(b_n[1]), float(b_n[2]), w, h)
            p1dir = self._project_norm_to_window(float(tip_n[0]), float(tip_n[1]), float(tip_n[2]), w, h)
            if p0 is None:
                continue
            x0, y0 = p0
            if p1dir is None:
                ux, uy = 1.0, 0.0
            else:
                dx, dy = p1dir[0] - x0, p1dir[1] - y0
                ln = float(math.hypot(dx, dy))
                if ln < 1e-6:
                    ux, uy = 1.0, 0.0
                else:
                    ux, uy = dx / ln, dy / ln

            frac = mag_abs / mag_max if mag_max > 0 else 1.0
            seg_len = min_px + (max_px - min_px) * min(1.0, frac)
            x1 = x0 + ux * seg_len
            y1 = y0 + uy * seg_len
            segments.append((x0, y0, x1, y1))

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        if not segments:
            return

        self._enter_ortho(w, h)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_1D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for x0, y0, x1, y1 in segments:
            glLineWidth(4.0)
            glColor4f(1.0, 1.0, 1.0, 1.0)
            self._draw_screen_arrow_2d(x0, y0, x1, y1, head_len=12.0, head_half=7.0)
        glLineWidth(1.0)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        self._leave_ortho()

    def _render_pick_scene(self, w: int, h: int) -> None:
        glClearColor(0.0, 0.0, 0.0, 1.0)  # id=0 = empty
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._setup_camera(w, h)

        glDisable(GL_LIGHTING)
        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_1D)
        glDisable(GL_TEXTURE_2D)
        glShadeModel(GL_SMOOTH)

        # Mesh (per-triangle IDs).
        if self._pick_mesh_verts is not None and self._pick_mesh_colors is not None:
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, self._pick_mesh_verts)
            glColorPointer(3, GL_UNSIGNED_BYTE, 0, self._pick_mesh_colors)
            n_verts = self._pick_mesh_verts.size // 3
            glDrawArrays(GL_TRIANGLES, 0, int(n_verts))
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)

        # Spheres - one draw per electrode (cheap with ~hundreds of points).
        positions = self._active_elec_positions()
        if positions.shape[0] > 0:
            radius = self._current_sphere_radius()
            base_v = (_SPH_V.astype(np.float32) * radius)
            base_v_flat = np.ascontiguousarray(base_v).reshape(-1)
            base_t = _SPH_T.astype(np.uint32).reshape(-1)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, base_v_flat)
            for k, pos in enumerate(positions):
                pick_id = k + 1
                r = pick_id & 0xFF
                g = (pick_id >> 8) & 0xFF
                b = (pick_id >> 16) & 0xFF
                glColor3ub(int(r), int(g), int(b))
                glPushMatrix()
                glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
                glDrawElements(GL_TRIANGLES, int(base_t.size), GL_UNSIGNED_INT, base_t)
                glPopMatrix()
            glDisableClientState(GL_VERTEX_ARRAY)

        # Patch anchor pick targets (small spheres, not shown in color pass).
        if self._patch_preview_active:
            patch_pos = self._patch_anchor_positions_norm()
            if patch_pos.shape[0] > 0:
                n_sph, n_tris, _ = self._pick_pool_sizes()
                prad = max(0.008, self._current_sphere_radius() * 0.85)
                base_v = (_SPH_V.astype(np.float32) * prad)
                base_v_flat = np.ascontiguousarray(base_v).reshape(-1)
                base_t = _SPH_T.astype(np.uint32).reshape(-1)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, base_v_flat)
                for i, pos in enumerate(patch_pos):
                    pick_id = n_sph + n_tris + 1 + int(i)
                    r = pick_id & 0xFF
                    g = (pick_id >> 8) & 0xFF
                    b = (pick_id >> 16) & 0xFF
                    glColor3ub(int(r), int(g), int(b))
                    glPushMatrix()
                    glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
                    glDrawElements(GL_TRIANGLES, int(base_t.size), GL_UNSIGNED_INT, base_t)
                    glPopMatrix()
                glDisableClientState(GL_VERTEX_ARRAY)

        # Flush (not finish): ``glFinish`` stalls the CPU until the GPU drains
        # the whole queue — too heavy for a hover pick that runs ~30–60 Hz.
        glFlush()

    def _update_hover_from_pick_id(self, pick_id: int) -> None:
        kind, local = self._decode_pick_id(int(pick_id))
        self._hover_kind = kind
        self._hover_patch_anchor = None
        if kind == "sphere":
            self._hover_global_idx = (
                self._elec_global_idx[local] if 0 <= local < len(self._elec_global_idx) else None
            )
            self._hover_triangle = None
        elif kind == "patch_anchor":
            self._hover_global_idx = None
            self._hover_triangle = None
            self._hover_patch_anchor = int(local) if local >= 0 else None
        elif kind == "triangle":
            self._hover_global_idx = None
            self._hover_triangle = local
        else:
            self._hover_global_idx = None
            self._hover_triangle = None

    # ------------------------------------------------------- HUD overlay
    def _draw_hud_text(self, w: int, h: int) -> None:
        if self._text is None:
            self._text = TextRenderer(font_size=12)
        self._enter_ortho(w, h)
        glDisable(GL_DEPTH_TEST)

        # Top-left status bar.
        kind = self._hover_kind
        if kind == "sphere" and self._hover_global_idx is not None:
            label = ""
            try:
                k = self._elec_global_idx.index(self._hover_global_idx)
                label = self._elec_labels[k]
            except Exception:
                pass
            txt = f"electrode  {label}  (row #{self._hover_global_idx})"
            color = (255, 240, 120, 255)
        elif kind == "patch_anchor" and self._hover_patch_anchor is not None:
            ai = int(self._hover_patch_anchor)
            label = ""
            if 0 <= ai < len(self._patch_anchor_labels):
                label = str(self._patch_anchor_labels[ai])
            txt = f"patch point  {label}  (#{ai})"
            color = (180, 255, 200, 255)
        elif kind == "triangle" and self._hover_triangle is not None:
            txt = f"{self.mesh_name}  triangle #{self._hover_triangle}"
            color = (200, 235, 255, 255)
        else:
            txt = "empty"
            color = (200, 200, 210, 255)
        self._text.draw(txt, 8, 8, rgba=color, anchor="nw", px=13)

        # Floating cursor tooltip near the pointer when over something.
        if self._cursor_pos is not None and kind != "empty":
            cx, cy = self._cursor_pos
            self._text.draw(txt, cx + 14, cy + 10, rgba=color, anchor="nw", px=12)

        glEnable(GL_DEPTH_TEST)
        self._leave_ortho()

    # ----------------------------------------------------------- redraw
    def initgl(self) -> None:
        if not _GL_AVAILABLE:
            return
        glClearColor(*self._BG_RGB)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glShadeModel(GL_SMOOTH)
        # Quality hints: nicer perspective interpolation across triangles +
        # smoother line / polygon edges. These are advisory but most drivers
        # honor them with little to no perf hit.
        try:
            from OpenGL.GL import (
                GL_PERSPECTIVE_CORRECTION_HINT,
                GL_POINT_SMOOTH_HINT,
                GL_LINE_SMOOTH_HINT,
                GL_NICEST,
                glHint,
                glEnable as _glEnable,
                GL_MULTISAMPLE,
            )
            glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            try:
                _glEnable(GL_MULTISAMPLE)
            except Exception:
                pass
        except Exception:
            pass
        try:
            glReadBuffer(GL_BACK)
        except Exception:
            pass
        if not self._mesh_loaded:
            try:
                self._load_mesh()
            except Exception as exc:
                traceback.print_exc()
                print(f"[CartoMeshViewer] mesh load failed: {exc}")
                self._mesh_loaded = False

    def redraw(self) -> None:
        if not _GL_AVAILABLE:
            return
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        glViewport(0, 0, w, h)

        # 1) Pick pass (only when we have a pending pick request).
        if self._pick_request is not None and self._mesh_loaded:
            px, py = self._pick_request
            try:
                self._render_pick_scene(w, h)
                gl_y = h - 1 - int(py)
                raw = glReadPixels(int(px), int(gl_y), 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
                if isinstance(raw, bytes):
                    arr = np.frombuffer(raw, dtype=np.uint8)
                else:
                    arr = np.asarray(raw, dtype=np.uint8).ravel()
                if arr.size >= 3:
                    pick_id = pk.rgb_to_id(arr[:3])
                    self._update_hover_from_pick_id(pick_id)
                    self._last_pick_at = _time.monotonic()
                    self._last_pick_xy = (int(px), int(py))
                if self._pending_click:
                    self._handle_pick_click()
                    self._pending_click = False
            except Exception:
                traceback.print_exc()
            self._pick_request = None

        # 2) Color render pass.
        self._render_color_scene(w, h)

        # 3) HUD: colorbar + tooltip text.
        self._draw_colorbar_gl(w, h)
        self._draw_hud_text(w, h)

    # ----------------------------------------------------------- mouse
    def _handle_pick_click(self) -> None:
        # Snapshot the hover state before firing the callback. The app-side
        # callback can trigger a re-entrant redraw (select -> update_plot ->
        # mesh sync -> _request_redraw -> tkExpose -> redraw) which may run
        # a fresh pick pass and reset _hover_global_idx to None mid-call.
        kind = self._hover_kind
        gidx = self._hover_global_idx
        patch_ai = self._hover_patch_anchor
        if kind == "patch_anchor" and patch_ai is not None:
            ai = int(patch_ai)
            self._selected_patch_anchor = ai
            cb = self.on_pick_callback
            if callable(cb):
                label = ""
                if 0 <= ai < len(self._patch_anchor_labels):
                    label = str(self._patch_anchor_labels[ai])
                try:
                    cb("patch_anchor", ai, {"label": label})
                except Exception:
                    traceback.print_exc()
            return
        if kind != "sphere" or gidx is None:
            return
        gidx = int(gidx)
        self._selected_global_idx = gidx
        cb = self.on_pick_callback
        if callable(cb):
            try:
                cb("sphere", gidx, {"label": self._lookup_label(gidx)})
            except Exception:
                traceback.print_exc()

    def _lookup_label(self, gidx: int) -> str:
        try:
            k = self._elec_global_idx.index(int(gidx))
            return self._elec_labels[k]
        except Exception:
            return ""

    def _on_b1_press(self, event) -> None:
        self._drag_origin = (event.x, event.y)
        try:
            self.focus_set()
        except Exception:
            pass
        if self._hit_colorbar(event.x, event.y):
            self._drag_mode = "cb_move"
            return
        # If a recent hover-pick already covers this pixel, reuse it to skip
        # the synchronous pick render (which is the main source of click lag
        # in the 3D viewer). The hover pick fires every ~80 ms on motion so
        # the cached result is fresh when the user clicks something.
        if (
            self._last_pick_xy is not None
            and abs(event.x - self._last_pick_xy[0]) <= 3
            and abs(event.y - self._last_pick_xy[1]) <= 3
            and _time.monotonic() - self._last_pick_at < 0.25
        ):
            if self._hover_kind == "sphere":
                self._handle_pick_click()
                self._drag_mode = None
                self._request_redraw()  # just refresh the halo
                return
            # Hovering empty / triangle: skip the pick render too.
            self._drag_mode = "rotate"
            return
        # Fall back: synchronous pick (tkExpose -> redraw -> readPixel).
        self._pick_request = (event.x, event.y)
        self._pending_click = True
        self._request_redraw()
        if self._hover_kind == "sphere":
            self._drag_mode = None
        else:
            self._drag_mode = "rotate"

    def _on_b1_motion(self, event) -> None:
        if self._drag_origin is None or self._drag_mode is None:
            return
        dx = event.x - self._drag_origin[0]
        dy = event.y - self._drag_origin[1]
        self._drag_origin = (event.x, event.y)
        if self._drag_mode == "cb_move":
            self.cb_drag_dx += dx
            self.cb_drag_dy += dy
        elif self._drag_mode == "rotate":
            self._rot_y += dx * 0.5
            self._rot_x += dy * 0.5
        self._request_redraw()

    def _on_b1_release(self, _e) -> None:
        self._drag_mode = None
        self._drag_origin = None

    def _on_b3_press(self, event) -> None:
        self._drag_origin = (event.x, event.y)
        if self._hit_colorbar(event.x, event.y):
            self._drag_mode = None
            try:
                self._ctx_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self._ctx_menu.grab_release()
            return
        # Force a pick so we know whether the cursor is over a sphere.
        if self._mesh_loaded and self._elec_xyz_norm.shape[0] > 0:
            self._pick_request = (event.x, event.y)
            self._pending_click = False
            self._request_redraw()
        if self._hover_kind == "sphere":
            self._drag_mode = None
            try:
                self._sphere_ctx_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self._sphere_ctx_menu.grab_release()
            return
        self._drag_mode = "pan"

    def _on_b3_motion(self, event) -> None:
        if self._drag_origin is None or self._drag_mode != "pan":
            return
        dx = event.x - self._drag_origin[0]
        dy = event.y - self._drag_origin[1]
        self._drag_origin = (event.x, event.y)
        self._pan[0] += dx * 0.003 * self._zoom
        self._pan[1] -= dy * 0.003 * self._zoom
        self._request_redraw()

    def _on_b3_release(self, _e) -> None:
        self._drag_mode = None
        self._drag_origin = None

    def _on_wheel(self, event) -> None:
        factor = 0.9 if event.delta > 0 else 1.1
        self._zoom = float(np.clip(self._zoom * factor, 0.2, 20.0))
        self._request_redraw()

    def _on_wheel_linux(self, event) -> None:
        factor = 0.9 if event.num == 4 else 1.1
        self._zoom = float(np.clip(self._zoom * factor, 0.2, 20.0))
        self._request_redraw()

    def _on_motion(self, event) -> None:
        self._cursor_pos = (event.x, event.y)
        # While the user is mid-drag we don't want extra pick passes fighting
        # for the GL context - the rotate/pan redraw already covers visuals.
        if self._drag_mode in ("rotate", "pan", "cb_move"):
            return
        # ~40 Hz hover pick: id buffer is rendered every tick; ``glFlush`` in
        # the pick pass avoids a full GPU drain each time.
        if self._motion_after is not None:
            return
        try:
            self._motion_after = self.after(24, self._dispatch_hover_pick)
        except Exception:
            self._motion_after = None

    def _dispatch_hover_pick(self) -> None:
        self._motion_after = None
        if self._cursor_pos is None or not self._mesh_loaded:
            return
        # Skip if cursor hasn't moved since the last successful pick.
        if (
            self._last_pick_xy is not None
            and self._last_pick_xy == self._cursor_pos
            and _time.monotonic() - self._last_pick_at < 0.5
        ):
            return
        self._pick_request = self._cursor_pos
        self._pending_click = False
        self._request_redraw()

    def _on_leave(self, _e) -> None:
        self._cursor_pos = None
        self._hover_kind = "empty"
        self._hover_global_idx = None
        self._hover_triangle = None
        self._request_redraw()

    # ----------------------------------------------------------- misc
    def _request_redraw(self) -> None:
        # Coalesce bursts of redraw requests (hover + mesh sync) so each event
        # loop tick does at most one GL flush instead of N synchronous exposes.
        if self._redraw_after_id is not None:
            try:
                self.after_cancel(self._redraw_after_id)
            except Exception:
                pass
            self._redraw_after_id = None

        def _do() -> None:
            self._redraw_after_id = None
            try:
                self.tkExpose(None)
            except Exception:
                pass

        try:
            self._redraw_after_id = self.after(12, _do)
        except Exception:
            self._redraw_after_id = None
            _do()

    def add_color_listener(self, cb) -> None:
        if cb not in self._color_listeners:
            self._color_listeners.append(cb)

    def _notify_color_listeners(self) -> None:
        for cb in list(self._color_listeners):
            try:
                cb(self)
            except Exception:
                traceback.print_exc()

    def open_colorbar_settings(self) -> None:
        from .ui import ColorbarSettingsDialog

        try:
            ColorbarSettingsDialog(self)
        except Exception:
            traceback.print_exc()
            from tkinter import messagebox as mb

            mb.showerror("Colorbar", "Could not open settings window.")

    def open_sphere_settings(self) -> None:
        from .ui import SphereSettingsDialog

        try:
            SphereSettingsDialog(self)
        except Exception:
            traceback.print_exc()
            from tkinter import messagebox as mb

            mb.showerror("Spheres", "Could not open settings window.")

    # ------------------------------------------------------- sphere knobs
    def set_sphere_radius_factor(self, factor: float) -> None:
        try:
            f = float(factor)
        except (TypeError, ValueError):
            return
        f = float(np.clip(f, 0.05, 8.0))
        if abs(f - self.sphere_radius_factor) < 1e-6:
            return
        self.sphere_radius_factor = f
        self._request_redraw()

    def _bump_radius_factor(self, mult: float) -> None:
        self.set_sphere_radius_factor(self.sphere_radius_factor * float(mult))

    def set_use_projected(self, flag: bool) -> None:
        new = bool(flag)
        if new == self.use_projected:
            return
        self.use_projected = new
        self._request_redraw()

    def toggle_projection(self) -> None:
        self.set_use_projected(not self.use_projected)

    def recompute_projection(self) -> None:
        if self._mesh_loaded:
            self._compute_projected_positions()
            self._request_redraw()

    def set_scalar_field(self, field: str) -> None:
        if self._patch_preview_active:
            return
        if field == self.scalar_field:
            return
        if str(field).startswith("delta:"):
            if not self.interpolation_enabled or self._delta_provider is None:
                return
            self.scalar_field = field
            self._current_delta_metric = field[len("delta:") :]
            if self.auto_range:
                self.vmin = None
                self.vmax = None
            if self._mesh_loaded:
                self._compute_delta_interpolated()
                self._request_redraw()
            return

        if field not in cm.SCALAR_FIELDS:
            return
        self._current_delta_metric = None
        self.scalar_field = field
        if self.auto_range:
            self.vmin = None
            self.vmax = None
        try:
            self._load_mesh()
            self._request_redraw()
        except Exception as exc:
            traceback.print_exc()
            print(f"[CartoMeshViewer] failed to switch scalar to {field}: {exc}")

    def apply_color_settings(
        self,
        cmap_name: str | None = None,
        reverse: bool | None = None,
        n_bins: int | None = None,
        auto_range: bool | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        color_mode: str | None = None,
        piece_knots: list[float] | None = None,
        custom_bins: list[dict] | None = None,
        cb_orientation: str | None = None,
        show_colorbar: bool | None = None,
    ) -> None:
        if cmap_name is not None:
            if cmap_name.endswith("_r"):
                self.cmap_name = cmap_name[:-2]
                self.reverse_cmap = True
            else:
                self.cmap_name = cmap_name
        if reverse is not None:
            self.reverse_cmap = bool(reverse)
        if n_bins is not None:
            self.n_bins = int(max(1, min(256, n_bins)))
        if auto_range is not None:
            self.auto_range = bool(auto_range)
        if vmin is not None:
            self.vmin = float(vmin)
        if vmax is not None:
            self.vmax = float(vmax)
        if color_mode in ("standard", "custom"):
            self.color_mode = color_mode
        if piece_knots is not None:
            self.piece_knots = cm.merge_knots01(list(piece_knots))
        if custom_bins is not None:
            self.custom_bins = copy.deepcopy(custom_bins)[: cm.MAX_CUSTOM_BINS]
        if cb_orientation in ("horizontal", "vertical"):
            self.cb_orientation = cb_orientation
        if show_colorbar is not None:
            self.show_colorbar = bool(show_colorbar)
        if self._scalars is None:
            return
        self._recompute_colors()
        self._request_redraw()

    def snapshot_color_settings(self) -> dict:
        smin, smax = self._resolved_range()
        return {
            "cmap_name": self.cmap_name,
            "reverse": self.reverse_cmap,
            "n_bins": self.n_bins,
            "auto_range": self.auto_range,
            "vmin": self.vmin if self.vmin is not None else smin,
            "vmax": self.vmax if self.vmax is not None else smax,
            "scalar_field": self.scalar_field,
            "color_mode": self.color_mode,
            "piece_knots": list(self.piece_knots),
            "custom_bins": copy.deepcopy(self.custom_bins),
            "cb_orientation": self.cb_orientation,
            "show_colorbar": self.show_colorbar,
            "cb_drag_dx": self.cb_drag_dx,
            "cb_drag_dy": self.cb_drag_dy,
        }

    def restore_color_settings(self, snap: dict) -> None:
        self.apply_color_settings(
            cmap_name=snap["cmap_name"],
            reverse=snap["reverse"],
            n_bins=snap["n_bins"],
            auto_range=snap["auto_range"],
            vmin=None if snap["auto_range"] else snap["vmin"],
            vmax=None if snap["auto_range"] else snap["vmax"],
            color_mode=snap.get("color_mode", "standard"),
            piece_knots=snap.get("piece_knots", []),
            custom_bins=snap.get("custom_bins", []),
            cb_orientation=snap.get("cb_orientation", "vertical"),
            show_colorbar=snap.get("show_colorbar", True),
        )
        self.cb_drag_dx = int(snap.get("cb_drag_dx", 0))
        self.cb_drag_dy = int(snap.get("cb_drag_dy", 0))

    def get_color_state(self) -> dict:
        smin, smax = self._resolved_range()
        return {
            "cmap_name": self.cmap_name,
            "reverse": self.reverse_cmap,
            "n_bins": self.n_bins,
            "auto_range": self.auto_range,
            "vmin": self.vmin if self.vmin is not None else smin,
            "vmax": self.vmax if self.vmax is not None else smax,
            "scalar_field": self.scalar_field,
            "resolved_vmin": smin,
            "resolved_vmax": smax,
            "color_mode": self.color_mode,
            "piece_knots": list(self.piece_knots),
            "custom_bins": copy.deepcopy(self.custom_bins),
        }

    def get_data_range(self) -> tuple[float, float]:
        if self._scalars is None or self._scalars.size == 0:
            return 0.0, 1.0
        finite = self._scalars[np.isfinite(self._scalars)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.min(finite)), float(np.max(finite))

    def reset_view(self) -> None:
        self._rot_x = 20.0
        self._rot_y = -30.0
        self._zoom = 2.8
        self._pan = [0.0, 0.0]
        self._request_redraw()

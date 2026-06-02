"""Per-patch (and legacy global) dv/ds pipeline.

A **patch** is one acquisition take/section (``carto.cont[i]``): a set of
electrodes whose unipolar signals were recorded simultaneously on the same
2.5 s timebase. :func:`compute_patches` builds, **per take** and per
pacing/sinus window type (SR, S1, S2, S3), a local spatio-temporal map of
the spatial conduction gradient ``|dV/ds|`` of the unipolar voltage over
just that take's footprint on the mesh.

Per-patch pipeline (see :func:`compute_patches`):

1. Group every non-rejected electrode point by its take/section.
2. For each take and window type, take that take's electrodes as the
   interpolation anchors (all sharing the take's reference/timebase) and
   re-base each window so ``t = 0`` is its reference sample; the shared
   relative-time axis is the inclusive union of the re-based windows.
3. Size the harmonic patch from the take's own electrode spacing so it
   covers all of the take's electrodes plus a small margin ("area around
   it"), instead of a fixed radius.
4. At every relative-time sample, harmonically interpolate the unipolar
   voltage within that patch (Neumann rim) and compute the per-vertex
   spatial gradient magnitude ``|dV/ds|``. Only the patch vertices are
   stored (the rest of the mesh stays blank/NaN).

The per-sample ``|dV/ds|`` patch fields are cached so a UI slider can scrub
time instantly on the 3D mesh; as the user navigates points, the take that
owns the current point becomes the active patch.

:func:`compute_global_patch` (the earlier whole-mesh union-patch variant) is
retained for reference but is no longer wired to the UI.

Reusable channel-resolution / reference helpers were migrated here from the
former ``ui/acquisition_patch_tool.py``.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .viewer3d import cv as cvmod
from .viewer3d import laplacian as lap

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .main_app import App


# Window types in display order. SR pulls from the sinus window/reference;
# S1/S2/S3 pull from stim window/reference index 0/1/2.
WINDOW_TYPES: tuple[str, ...] = ("SR", "S1", "S2", "S3")

# Default acquisition sampling rate (Carto exports at 1 kHz; the signal index
# is in seconds so 1 sample == 1 ms). Used only as a fallback when the index
# spacing can't be inferred.
DEFAULT_FS = 1000.0


# --------------------------------------------------------------------------
# Channel-resolution helpers (migrated from ui/acquisition_patch_tool.py)
# --------------------------------------------------------------------------
def _resolve_uni_col(signals, name: str) -> str:
    from .plotting.presenter import PlotPresenter

    return PlotPresenter._resolve_channel_name(signals, str(name))


def _resolve_ref_col(signals, ref_channel: str | None) -> str | None:
    from .plotting.presenter import PlotPresenter

    try:
        return PlotPresenter._resolve_first_existing(
            signals, ["V5", ref_channel, "CS1", "M4", "M3"]
        )
    except Exception:
        return None


def _pad_range(vmin: float, vmax: float) -> tuple[float, float]:
    """Symmetric 2% padding for a colorbar range; safe defaults if degenerate."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return 0.0, 1.0
    pad = 0.02 * (vmax - vmin)
    return vmin - pad, vmax + pad


def _infer_fs(signals) -> float:
    """Sampling rate (Hz) inferred from the signal index spacing in seconds."""
    try:
        idx = np.asarray(signals.index, dtype=np.float64).ravel()
    except Exception:
        return DEFAULT_FS
    if idx.size < 2:
        return DEFAULT_FS
    dt = np.median(np.diff(idx))
    if not np.isfinite(dt) or dt <= 0:
        return DEFAULT_FS
    fs = 1.0 / dt
    # Guard against an index already expressed in milliseconds (dt ~ 1).
    if fs < 10.0:
        return DEFAULT_FS
    return float(fs)


# --------------------------------------------------------------------------
# Window extraction
# --------------------------------------------------------------------------
def _window_ref_and_span(c1: dict, window_type: str) -> tuple[int, int, int] | None:
    """Return ``(ref_sample, start_sample, end_sample)`` for one window type.

    ``c1`` is the per-point metrics dict stored in ``app.delta[gidx][2]``.
    Returns ``None`` when the requested window is missing / not measurable
    (a ``False`` window marker or an out-of-range stim index).
    """
    if not isinstance(c1, dict):
        return None
    if window_type == "SR":
        refs = c1.get("refs_sinus") or []
        wins = c1.get("sinus") or []
        wi = 0
    else:
        try:
            wi = int(window_type[1:]) - 1  # "S1" -> 0
        except (ValueError, IndexError):
            return None
        refs = c1.get("refs_stim") or []
        wins = c1.get("stim") or []
    if wi < 0 or wi >= len(wins) or wi >= len(refs):
        return None
    win = wins[wi]
    if not (isinstance(win, (list, tuple)) and len(win) == 2):
        return None
    try:
        start = int(win[0])
        end = int(win[1])
        ref = int(refs[wi])
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return ref, start, end


def gather_global_anchors(app: "App", window_type: str) -> list[dict]:
    """Collect per-point anchor data for one window type.

    Each returned dict has::

        {
            "global_idx": int,
            "pos": (x, y, z),
            "trace": np.ndarray,    # full unipolar trace (mV)
            "ref": int,             # reference sample index
            "lo": int,              # start - ref  (re-based, samples)
            "hi": int,              # end   - ref  (re-based, samples)
            "fs": float,
        }
    """
    anchors: list[dict] = []
    delta = getattr(app, "delta", None) or []
    to_i_j = getattr(app, "to_i_j", None) or []
    carto = app.carto
    for gidx, entry in enumerate(delta):
        if gidx >= len(to_i_j):
            break
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
            continue
        label = str(entry[1] or "").strip().lower()
        if label == "reject":
            continue
        c1 = entry[2]
        span = _window_ref_and_span(c1, window_type)
        if span is None:
            continue
        ref, start, end = span
        i, j = to_i_j[gidx]
        try:
            section = carto.cont[int(i)]
            df_pt = section[0]
            signals = section[2]
            row = df_pt.iloc[int(j)]
        except Exception:
            continue
        try:
            cname = _resolve_uni_col(signals, str(row["unipolar"]))
            trace = np.asarray(signals[cname].values, dtype=np.float64)
        except Exception:
            continue
        try:
            pos = (float(row["x"]), float(row["y"]), float(row["z"]))
        except Exception:
            continue
        if not all(np.isfinite(pos)):
            continue
        if trace.size == 0 or not (0 <= ref < trace.size):
            continue
        anchors.append(
            {
                "global_idx": int(gidx),
                "pos": pos,
                "trace": trace,
                "ref": int(ref),
                "lo": int(start - ref),
                "hi": int(end - ref),
                "fs": _infer_fs(signals),
            }
        )
    return anchors


def count_valid_windows(c1: dict) -> tuple[int, int]:
    """Return ``(n_valid_stim, n_valid_sinus)`` measurable windows in ``c1``.

    A window counts only when it is a concrete ``[start, end]`` pair (the
    presenter writes ``False`` for stims/sinus it could not measure).
    """
    if not isinstance(c1, dict):
        return 0, 0

    def _n(key: str) -> int:
        wins = c1.get(key) or []
        return sum(1 for w in wins if isinstance(w, (list, tuple)) and len(w) == 2)

    return _n("stim"), _n("sinus")


def build_shared_axis(anchors: list[dict]) -> np.ndarray | None:
    """Inclusive (union) relative-time axis as integer sample offsets.

    Each point's window of interest is re-based so ``t = 0`` is its own
    reference; the shared axis spans the union ``[min(lo), max(hi)]`` of all
    participating windows. Using the union (rather than a strict
    intersection) is required because each point's activation window sits at
    a different delay relative to the reference - the whole point of LAT
    mapping - so a strict intersection routinely collapses to nothing. Each
    point still contributes its unipolar voltage at every shared sample
    (sampled from its full trace, clamped at the trace bounds).

    Returns ``np.arange(lo, hi + 1)`` or ``None`` if degenerate.
    """
    if not anchors:
        return None
    lo = min(int(a["lo"]) for a in anchors)
    hi = max(int(a["hi"]) for a in anchors)
    if hi <= lo:
        return None
    return np.arange(lo, hi + 1, dtype=np.int64)


# --------------------------------------------------------------------------
# Geodesic distances + pre-factorized global-patch harmonic solver
# --------------------------------------------------------------------------
def _geodesic_distances(
    graph, V: np.ndarray, F: np.ndarray, anchor_vidx: np.ndarray, radius: float
) -> np.ndarray:
    """``(n_anchors, n_vertices)`` geodesic distances (inf beyond ``radius``).

    Prefers edge-Dijkstra (robust, always available); falls back to FMM then
    the heat method, mirroring the viewer's interpolation worker.
    """
    try:
        d = lap.dijkstra_distances_from_anchors(graph, anchor_vidx, radius)
        if d.size and np.isfinite(d).any():
            return d
    except Exception:
        traceback.print_exc()
    try:
        d = lap.fmm_distances_from_anchors(V, F, anchor_vidx, radius)
        if d.size and np.isfinite(d).any():
            return d
    except Exception:
        traceback.print_exc()
    return lap.heat_geodesic_distances_from_anchors(
        V, F, anchor_vidx, radius, tau_reference_radius=radius
    )


class _GlobalPatchSolver:
    """Pre-factorized global-patch (Neumann-rim) harmonic solver.

    The patch (union of geodesic balls) and the cotangent Laplacian are
    fixed across all time samples, so the Neumann sub-Laplacian is built and
    LU-factorized once; each :meth:`solve` only re-evaluates the right-hand
    side for new anchor values. Mirrors ``laplacian._solve_neumann_patch``
    with duplicate-vertex anchors averaged.
    """

    def __init__(
        self,
        L: sp.spmatrix,
        dist: np.ndarray,
        anchor_vidx: np.ndarray,
        radius: float,
    ) -> None:
        self.n = int(L.shape[0])
        raw = np.asarray(anchor_vidx, dtype=np.int64).ravel()
        # Average electrodes that snap to the same mesh vertex.
        unique, inverse = np.unique(raw, return_inverse=True)
        self.anchor_idx = unique
        self._inverse = inverse
        self._n_unique = int(unique.size)

        reach = np.asarray(dist, dtype=np.float64) <= float(radius)
        n_reach = reach.sum(axis=0)
        is_anchor = np.zeros(self.n, dtype=bool)
        is_anchor[self.anchor_idx] = True
        in_patch = n_reach >= 1
        self.free = np.where(in_patch & ~is_anchor)[0]

        self._lu = None
        self._A_FF = None
        self._A_FA = None
        if self.free.size:
            L_csc = L.tocsc()
            A_FF = L_csc[self.free, :][:, self.free].tolil()
            A_FA = L_csc[self.free, :][:, self.anchor_idx]
            ff_off = np.asarray(A_FF.sum(axis=1)).ravel() - A_FF.diagonal()
            fa_row = np.asarray(A_FA.sum(axis=1)).ravel()
            A_FF.setdiag(-(ff_off + fa_row))
            self._A_FF = A_FF.tocsc()
            self._A_FA = A_FA.tocsc()
            try:
                self._lu = spla.splu(self._A_FF)
            except Exception:
                traceback.print_exc()
                self._lu = None

    def _dedup_values(self, anchor_val_raw: np.ndarray) -> np.ndarray:
        av = np.asarray(anchor_val_raw, dtype=np.float64).ravel()
        if av.size == self._n_unique and self._n_unique == self._inverse.size:
            # No duplicates: raw already aligns 1:1 with unique anchors.
            if np.array_equal(self._inverse, np.arange(self._n_unique)):
                return av
        sums = np.zeros(self._n_unique, dtype=np.float64)
        counts = np.zeros(self._n_unique, dtype=np.int64)
        np.add.at(sums, self._inverse, av)
        np.add.at(counts, self._inverse, 1)
        return sums / np.maximum(counts, 1)

    def solve(self, anchor_val_raw: np.ndarray) -> np.ndarray:
        """Per-vertex harmonic field; NaN outside the patch."""
        f = np.full(self.n, np.nan, dtype=np.float64)
        av = self._dedup_values(anchor_val_raw)
        f[self.anchor_idx] = av
        if self.free.size and self._A_FA is not None:
            rhs = -np.asarray(self._A_FA @ av).ravel()
            if self._lu is not None:
                try:
                    f_F = self._lu.solve(rhs)
                except Exception:
                    f_F, *_ = spla.lsqr(self._A_FF, rhs)
            else:
                f_F, *_ = spla.lsqr(self._A_FF, rhs)
            f[self.free] = f_F
        return f


# --------------------------------------------------------------------------
# Top-level driver
# --------------------------------------------------------------------------
def compute_global_patch(
    app: "App",
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Compute per-window-type global ``|dV/ds|`` spatio-temporal maps.

    Returns ``{window_type: result}`` where each ``result`` is::

        {
            "t_rel":        np.ndarray,            # relative time (ms from ref)
            "dvds_series":  np.ndarray float32,    # (n_samples, n_vertices)
            "vmin", "vmax": float,                 # fixed colour range
            "anchor_vertex": dict[int, int],       # global_idx -> mesh vertex
            "radius":       float,
            "n_anchors":    int,
        }

    ``progress_cb(done, total, message)`` is called from the worker thread;
    the caller is responsible for marshalling UI updates to the Tk loop.

    Window types with fewer than two usable points are skipped and absent
    from the result.
    """
    carto = app.carto
    V = np.asarray(carto.vertices, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(carto.triangles, dtype=np.int64).reshape(-1, 3)
    if V.shape[0] == 0 or F.shape[0] == 0:
        raise RuntimeError("Mesh vertices/triangles are not available on the Carto object.")

    mean_edge = lap.mean_edge_length(V, F)
    radius = 10.0 * float(mean_edge) if mean_edge > 0 else 0.0
    if radius <= 0:
        raise RuntimeError("Could not determine a positive interpolation radius (mean edge length is zero).")

    # Phase A: gather anchors + relative-time axis for every window type.
    prepared: dict[str, dict] = {}
    for wt in WINDOW_TYPES:
        anchors = gather_global_anchors(app, wt)
        if len(anchors) < 2:
            continue
        rel = build_shared_axis(anchors)
        if rel is None or rel.size == 0:
            continue
        fs = float(anchors[0].get("fs", DEFAULT_FS)) or DEFAULT_FS
        prepared[wt] = {"anchors": anchors, "rel": rel, "fs": fs}

    if not prepared:
        return {}

    total_samples = sum(int(p["rel"].size) for p in prepared.values())
    done = 0

    # Shared mesh operators (built once).
    L = lap.cot_laplacian(V, F)
    graph = lap.build_mesh_graph(V, F)

    results: dict[str, dict] = {}
    n_v = V.shape[0]

    for wt, prep in prepared.items():
        anchors = prep["anchors"]
        rel = prep["rel"]
        fs = prep["fs"]
        n_samples = int(rel.size)

        pts = np.asarray([a["pos"] for a in anchors], dtype=np.float64)
        anc_vidx = lap.map_points_to_vertices(V, pts)

        # Sample each point's unipolar trace on the shared relative-sample
        # grid: absolute sample = ref + offset (clamped into the trace).
        S = np.empty((n_samples, len(anchors)), dtype=np.float64)
        for ai, a in enumerate(anchors):
            tr = a["trace"]
            idx = np.clip(a["ref"] + rel, 0, tr.size - 1)
            S[:, ai] = tr[idx]

        try:
            dist = _geodesic_distances(graph, V, F, anc_vidx, radius)
            solver = _GlobalPatchSolver(L, dist, anc_vidx, radius)
        except Exception:
            traceback.print_exc()
            done += n_samples
            continue

        dvds_series = np.full((n_samples, n_v), np.nan, dtype=np.float32)
        vmin, vmax = np.inf, -np.inf
        for k in range(n_samples):
            if progress_cb is not None:
                progress_cb(done, total_samples, f"{wt}: dV/ds field {k + 1}/{n_samples}")
            phi = solver.solve(S[k, :])
            dvds = cvmod.vertex_gradient_magnitude(V, F, phi)
            dvds_series[k] = dvds.astype(np.float32)
            fin = dvds[np.isfinite(dvds)]
            if fin.size:
                vmin = min(vmin, float(np.min(fin)))
                vmax = max(vmax, float(np.max(fin)))
            done += 1

        lo_pad, hi_pad = _pad_range(vmin, vmax)
        anchor_vertex = {
            int(a["global_idx"]): int(anc_vidx[ai]) for ai, a in enumerate(anchors)
        }
        results[wt] = {
            "t_rel": (rel.astype(np.float64) * (1000.0 / fs)),
            "dvds_series": dvds_series,
            "vmin": float(lo_pad),
            "vmax": float(hi_pad),
            "anchor_vertex": anchor_vertex,
            "radius": float(radius),
            "n_anchors": int(len(anchors)),
        }

    if progress_cb is not None:
        progress_cb(total_samples, total_samples, "Done")
    return results


# --------------------------------------------------------------------------
# Per-patch (per take/section) driver
# --------------------------------------------------------------------------
# A take's electrodes can sit far apart on a curved surface, so the geodesic
# distance used to build a patch is computed out to a generous cap derived
# from the take's straight-line electrode spread.
_GEO_CAP_SPREAD_FACTOR = 1.5
_GEO_CAP_EDGE_MARGIN = 5.0

# A patch must hold at least this many electrodes for a spatial gradient to
# be meaningful (two points only define a line / a degenerate field).
_MIN_PATCH_ANCHORS = 3


def _anchor_from_point(c1, wt, gidx, pos, trace, fs):
    """Build one anchor for a window type, or ``None`` when unusable.

    The interpolation window is the **window of interest** measured from the
    window's reference (``t = 0`` at the reference) through its end - NOT the
    narrow activation sub-window. So every electrode's window starts strictly
    at its reference (``lo = 0``) and runs to ``end - ref``.
    """
    span = _window_ref_and_span(c1, wt)
    if span is None:
        return None
    ref, _start, end = span
    if not (0 <= ref < trace.size):
        return None
    hi = int(end - ref)
    if hi <= 0:
        return None
    return {
        "global_idx": int(gidx),
        "pos": pos,
        "trace": trace,
        "ref": int(ref),
        "lo": 0,
        "hi": hi,
        "fs": fs,
    }


def gather_section_anchors(app: "App", section_i: int) -> dict[str, list[dict]]:
    """Anchors for a single take/section, grouped by window type.

    Returns ``{window_type: [anchor, ...]}`` over just that take's points
    (skipping ``reject``). Iterating one section keeps lazy per-patch
    computes cheap.
    """
    out: dict[str, list[dict]] = {}
    delta = getattr(app, "delta", None) or []
    to_index = getattr(app, "to_index", None) or []
    carto = app.carto
    si = int(section_i)
    if si < 0 or si >= len(to_index):
        return out
    try:
        section = carto.cont[si]
        df_pt = section[0]
        signals = section[2]
    except Exception:
        return out
    fs = _infer_fs(signals)
    for j, gidx in enumerate(to_index[si]):
        if gidx >= len(delta):
            continue
        entry = delta[gidx]
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
            continue
        if str(entry[1] or "").strip().lower() == "reject":
            continue
        c1 = entry[2]
        try:
            row = df_pt.iloc[int(j)]
            cname = _resolve_uni_col(signals, str(row["unipolar"]))
            trace = np.asarray(signals[cname].values, dtype=np.float64)
            pos = (float(row["x"]), float(row["y"]), float(row["z"]))
        except Exception:
            continue
        if not all(np.isfinite(pos)) or trace.size == 0:
            continue
        for wt in WINDOW_TYPES:
            a = _anchor_from_point(c1, wt, gidx, pos, trace, fs)
            if a is not None:
                out.setdefault(wt, []).append(a)
    return out


def gather_patch_anchors(app: "App") -> dict[int, dict[str, list[dict]]]:
    """Group anchor data by take/section then window type (all sections).

    Returns ``{section_i: {window_type: [anchor, ...]}}``; see
    :func:`gather_section_anchors` for the per-anchor schema.
    """
    out: dict[int, dict[str, list[dict]]] = {}
    to_index = getattr(app, "to_index", None) or []
    for si in range(len(to_index)):
        bywt = gather_section_anchors(app, si)
        if bywt:
            out[int(si)] = bywt
    return out


def build_shared_ops(app: "App") -> dict:
    """Build the mesh operators shared across every patch (once per mesh).

    The cotangent Laplacian and edge graph over the whole mesh are the
    expensive one-time cost; cache the returned dict and reuse it for every
    lazy per-take compute.
    """
    carto = app.carto
    V = np.asarray(carto.vertices, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(carto.triangles, dtype=np.int64).reshape(-1, 3)
    if V.shape[0] == 0 or F.shape[0] == 0:
        raise RuntimeError("Mesh vertices/triangles are not available on the Carto object.")
    mean_edge = lap.mean_edge_length(V, F)
    if not np.isfinite(mean_edge) or mean_edge <= 0:
        raise RuntimeError("Could not determine a positive mean edge length on the mesh.")
    L = lap.cot_laplacian(V, F)
    graph = lap.build_mesh_graph(V, F)
    return {"V": V, "F": F, "L": L, "graph": graph, "mean_edge": float(mean_edge)}


def _build_patch_result(
    ops, anchors, rel, fs, sec, wt, progress_cb=None, done=0, total=0
):
    """Interpolate + |dV/ds| for one take/window patch.

    Returns ``(result_dict_or_None, done)``. The spatial gradient is
    evaluated on **only the patch's incident faces** (not the whole mesh),
    which is the key speed-up over the global pipeline.
    """
    V = ops["V"]
    F = ops["F"]
    L = ops["L"]
    graph = ops["graph"]
    mean_edge = ops["mean_edge"]
    n_v = V.shape[0]
    n_samples = int(rel.size)

    pts = np.asarray([a["pos"] for a in anchors], dtype=np.float64)
    anc_vidx = lap.map_points_to_vertices(V, pts)

    # Geodesic cap from the take's straight-line electrode spread.
    if pts.shape[0] >= 2:
        diff = pts[:, None, :] - pts[None, :, :]
        spread = float(np.max(np.linalg.norm(diff, axis=2)))
    else:
        spread = 0.0
    cap = _GEO_CAP_SPREAD_FACTOR * spread + _GEO_CAP_EDGE_MARGIN * float(mean_edge)
    if not np.isfinite(cap) or cap <= 0:
        cap = 10.0 * float(mean_edge)

    dist = _geodesic_distances(graph, V, F, anc_vidx, cap)
    radius = _patch_radius_from_spacing(dist, anc_vidx, mean_edge)
    solver = _GlobalPatchSolver(L, dist, anc_vidx, radius)

    patch_vertices = np.where((np.asarray(dist) <= radius).any(axis=0))[0]
    if patch_vertices.size == 0:
        return None, done + n_samples
    patch_vertices = np.sort(patch_vertices.astype(np.int64))

    # Faces incident to the patch — restricts the per-sample gradient.
    inpatch = np.zeros(n_v, dtype=bool)
    inpatch[patch_vertices] = True
    patch_faces = F[inpatch[F].any(axis=1)]

    # Sample each electrode's unipolar trace on the shared relative grid.
    S = np.empty((n_samples, len(anchors)), dtype=np.float64)
    for ai, a in enumerate(anchors):
        tr = a["trace"]
        idx = np.clip(a["ref"] + rel, 0, tr.size - 1)
        S[:, ai] = tr[idx]

    dvds_patch = np.full((n_samples, patch_vertices.size), np.nan, dtype=np.float32)
    vmin, vmax = np.inf, -np.inf
    for k in range(n_samples):
        if progress_cb is not None:
            progress_cb(done, total, f"take {sec} {wt}: dV/ds field {k + 1}/{n_samples}")
        phi = solver.solve(S[k, :])
        dvds = cvmod.vertex_gradient_magnitude(V, patch_faces, phi)
        col = dvds[patch_vertices].astype(np.float32)
        dvds_patch[k] = col
        fin = col[np.isfinite(col)]
        if fin.size:
            vmin = min(vmin, float(np.min(fin)))
            vmax = max(vmax, float(np.max(fin)))
        done += 1

    lo_pad, hi_pad = _pad_range(vmin, vmax)
    anchor_vertex = {
        int(a["global_idx"]): int(anc_vidx[ai]) for ai, a in enumerate(anchors)
    }
    ref_repr = int(np.median([int(a["ref"]) for a in anchors]))
    result = {
        "rel": rel.astype(np.int64),
        "fs": float(fs),
        "ref_repr": ref_repr,
        "patch_vertices": patch_vertices,
        "dvds_patch": dvds_patch,
        "vmin": float(lo_pad),
        "vmax": float(hi_pad),
        "anchor_vertex": anchor_vertex,
        "radius": float(radius),
        "n_anchors": int(len(anchors)),
    }
    return result, done


def compute_section_patches(
    app: "App",
    section_i: int,
    ops: dict | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, dict]:
    """Compute (and return) the per-window patch maps for ONE take/section.

    ``{window_type: result}`` (see :func:`compute_patches` for the result
    schema). ``ops`` should be the cached :func:`build_shared_ops` output so
    repeated per-take calls only pay the local Dijkstra + solves. Windows
    with fewer than :data:`_MIN_PATCH_ANCHORS` usable points are skipped.
    """
    if ops is None:
        ops = build_shared_ops(app)

    bywt = gather_section_anchors(app, section_i)
    tasks: list[tuple[str, list[dict], np.ndarray, float]] = []
    for wt in WINDOW_TYPES:
        anchors = bywt.get(wt) or []
        if len(anchors) < _MIN_PATCH_ANCHORS:
            continue
        rel = build_shared_axis(anchors)
        if rel is None or rel.size == 0:
            continue
        fs = float(anchors[0].get("fs", DEFAULT_FS)) or DEFAULT_FS
        tasks.append((wt, anchors, rel, fs))

    if not tasks:
        return {}

    total = sum(int(rel.size) for _, _, rel, _ in tasks)
    done = 0
    out: dict[str, dict] = {}
    for wt, anchors, rel, fs in tasks:
        try:
            result, done = _build_patch_result(
                ops, anchors, rel, fs, section_i, wt, progress_cb, done, total
            )
        except Exception:
            traceback.print_exc()
            done += int(rel.size)
            continue
        if result is not None:
            out[wt] = result

    if progress_cb is not None:
        progress_cb(total, total, "Done")
    return out


def _patch_radius_from_spacing(
    dist: np.ndarray, anc_vidx: np.ndarray, mean_edge: float
) -> float:
    """Inclusion radius that covers all of a take's electrodes + a margin.

    ``dist`` is ``(n_anchors, n_vertices)`` geodesic distance (inf past the
    cap). The largest nearest-neighbour gap between the take's electrodes
    sets how far a per-electrode ball must reach for the balls to overlap
    into one connected patch; a small margin (a couple of mean edges, or
    half the median spacing) is added so the patch extends slightly past the
    outermost electrodes ("area around it").
    """
    sub = np.asarray(dist)[:, np.asarray(anc_vidx, dtype=np.int64)]
    sub = np.array(sub, dtype=np.float64)
    np.fill_diagonal(sub, np.inf)
    nn = np.min(sub, axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return 10.0 * float(mean_edge)
    gap = float(np.max(nn))
    margin = max(2.0 * float(mean_edge), 0.5 * float(np.median(nn)))
    radius = gap + margin
    if not np.isfinite(radius) or radius <= 0:
        radius = 10.0 * float(mean_edge)
    return radius


def compute_patches(
    app: "App",
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[int, dict[str, dict]]:
    """Compute per-take, per-window ``|dV/ds|`` spatio-temporal patch maps.

    Returns ``{section_i: {window_type: result}}`` where each ``result`` is::

        {
            "rel":            np.ndarray int,      # relative sample offsets
            "fs":             float,               # sampling rate (Hz)
            "ref_repr":       int,                 # representative ref sample
            "patch_vertices": np.ndarray int,      # mesh vertices in the patch
            "dvds_patch":     np.ndarray float32,  # (n_samples, n_patch_vertices)
            "vmin", "vmax":   float,               # fixed colour range
            "anchor_vertex":  dict[int, int],      # global_idx -> mesh vertex
            "radius":         float,
            "n_anchors":      int,
        }

    Takes/windows with fewer than :data:`_MIN_PATCH_ANCHORS` usable points
    are skipped. ``progress_cb(done, total, message)`` is called from the
    worker thread (samples summed across every take/window). Prefer the lazy
    :func:`compute_section_patches` for the interactive UI.
    """
    ops = build_shared_ops(app)
    grouped = gather_patch_anchors(app)

    tasks: list[tuple[int, str, list[dict], np.ndarray, float]] = []
    for sec in sorted(grouped):
        bywt = grouped[sec]
        for wt in WINDOW_TYPES:
            anchors = bywt.get(wt) or []
            if len(anchors) < _MIN_PATCH_ANCHORS:
                continue
            rel = build_shared_axis(anchors)
            if rel is None or rel.size == 0:
                continue
            fs = float(anchors[0].get("fs", DEFAULT_FS)) or DEFAULT_FS
            tasks.append((sec, wt, anchors, rel, fs))

    if not tasks:
        return {}

    total_samples = sum(int(rel.size) for _, _, _, rel, _ in tasks)
    done = 0
    results: dict[int, dict[str, dict]] = {}
    for sec, wt, anchors, rel, fs in tasks:
        try:
            result, done = _build_patch_result(
                ops, anchors, rel, fs, sec, wt, progress_cb, done, total_samples
            )
        except Exception:
            traceback.print_exc()
            done += int(rel.size)
            continue
        if result is not None:
            results.setdefault(int(sec), {})[wt] = result

    if progress_cb is not None:
        progress_cb(total_samples, total_samples, "Done")
    return results

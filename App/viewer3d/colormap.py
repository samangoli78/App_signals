"""Colormap names, knot remapping, and matplotlib LUT access for the 3D viewer.

This module is intentionally free of Tk/OpenGL so it stays easy to test and reuse.
"""

from __future__ import annotations

import numpy as np

try:
    from matplotlib import colormaps as _mpl_colormaps

    def get_cmap(name: str):
        return _mpl_colormaps.get_cmap(name)
except Exception:  # pragma: no cover
    from matplotlib import cm as _cm

    def get_cmap(name: str):
        return _cm.get_cmap(name)


SCALAR_FIELDS = ("bipolar", "unipolar", "LAT")

COLORMAPS: tuple[str, ...] = (
    "jet", "turbo", "rainbow", "nipy_spectral", "gist_rainbow",
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
    "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu",
    "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
    "hot", "afmhot", "gist_heat", "copper",
    "cool", "winter", "spring", "summer", "autumn",
    "gray", "bone", "pink",
    "coolwarm", "bwr", "seismic",
    "RdBu", "RdBu_r", "RdYlBu", "RdYlBu_r",
    "RdYlGn", "RdYlGn_r", "Spectral", "Spectral_r",
    "PiYG", "PRGn", "BrBG", "PuOr",
    "gist_earth", "terrain", "ocean",
    "twilight", "twilight_shifted", "hsv",
)

MAX_CUSTOM_BINS = 5
MAX_INTERNAL_KNOTS = 8


def default_custom_orange_green_palette(n: int) -> list[tuple[float, float, float]]:
    """Default custom-bin colours: orange (low) → green (high), left to right."""
    n = max(1, int(n))
    orange = (1.0, 0.55, 0.0)
    green = (0.15, 0.75, 0.20)
    if n == 1:
        return [orange]
    out: list[tuple[float, float, float]] = []
    for i in range(n):
        t = i / float(n - 1)
        out.append(
            (
                orange[0] * (1.0 - t) + green[0] * t,
                orange[1] * (1.0 - t) + green[1] * t,
                orange[2] * (1.0 - t) + green[2] * t,
            )
        )
    return out


def merge_knots01(knots: list[float]) -> list[float]:
    out: list[float] = []
    for x in sorted(knots):
        if x <= 1e-6 or x >= 1.0 - 1e-6:
            continue
        if out and abs(x - out[-1]) < 1e-4:
            continue
        out.append(float(x))
    return out[:MAX_INTERNAL_KNOTS]


def full_knot_array(internal: list[float]) -> np.ndarray:
    return np.array([0.0] + sorted(internal) + [1.0], dtype=np.float64)


def t_to_cmap_u(t: np.ndarray, internal_knots: list[float]) -> np.ndarray:
    """Remap normalized data ``t`` in [0,1] to colormap coordinate ``u`` in [0,1]."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    knots = full_knot_array(internal_knots)
    ns = knots.size - 1
    if ns < 1:
        return t
    idx = np.searchsorted(knots, t, side="right") - 1
    idx = np.clip(idx, 0, ns - 1)
    lo = knots[idx]
    hi = knots[idx + 1]
    span = hi - lo
    span = np.where(span < 1e-15, 1.0, span)
    alpha = (t - lo) / span
    alpha = np.clip(alpha, 0.0, 1.0)
    return (idx.astype(np.float64) + alpha) / float(ns)


def rgb_to_hex(rgb) -> str:
    r = int(round(float(rgb[0]) * 255))
    g = int(round(float(rgb[1]) * 255))
    b = int(round(float(rgb[2]) * 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def effective_cmap_name(cmap_name: str, reverse: bool) -> str:
    name = cmap_name
    if reverse and not name.endswith("_r"):
        name = name + "_r"
    return name


def build_1d_lut_rgb(
    *,
    lut_width: int,
    color_mode: str,
    cmap_name: str,
    reverse_cmap: bool,
    n_bins: int,
    piece_knots: list[float],
    custom_bins: list[dict],
    vmin: float,
    vmax: float,
) -> tuple[np.ndarray, bool]:
    """Return ``(rgb_uint8, use_nearest_filter)`` for a GL_TEXTURE_1D.

    Texel ``i`` corresponds to ``s = (i + 0.5) / lut_width`` (texture coordinate).
    For ``standard`` mode, ``s`` is the matplotlib colormap parameter after knots
    and global bin quantization (same path as per-vertex ``u``).
    For ``custom`` mode, ``s`` is linear in data space from ``vmin`` to ``vmax`` and
    the LUT is piecewise-constant → use nearest filtering on the GL side.
    """
    w = max(16, int(lut_width))
    s = (np.arange(w, dtype=np.float64) + 0.5) / w
    nearest = False
    if color_mode == "custom" and custom_bins:
        nearest = True
        if vmax - vmin < 1e-12:
            vals = np.full(w, vmin)
        else:
            vals = vmin + s * (vmax - vmin)
        out = np.zeros((w, 3), dtype=np.float64)
        bins = sorted(custom_bins, key=lambda b: float(b["lo"]))
        for j in range(w):
            v = vals[j]
            c = np.array(bins[0]["rgb"], dtype=np.float64)
            for b in bins:
                if float(b["lo"]) <= v <= float(b["hi"]):
                    c = np.array(b["rgb"], dtype=np.float64)
                    break
            else:
                if v < float(bins[0]["lo"]):
                    c = np.array(bins[0]["rgb"], dtype=np.float64)
                else:
                    c = np.array(bins[-1]["rgb"], dtype=np.float64)
            out[j] = c
    else:
        name = effective_cmap_name(cmap_name, reverse_cmap)
        try:
            cm = get_cmap(name)
        except Exception:
            cm = get_cmap("jet")
        u = np.asarray(s, dtype=np.float64)
        nb = max(1, int(n_bins))
        if nb < 256:
            lv = np.floor(u * nb).clip(0, nb - 1)
            u = (lv + 0.5) / nb
        out = cm(u)[:, :3]
    rgb = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb, nearest


def compute_vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from triangle soup (``verts`` Nx3, ``tris`` Mx3 int)."""
    n_v = verts.shape[0]
    acc = np.zeros((n_v, 3), dtype=np.float64)
    for tri in tris:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        fn = np.cross(v1 - v0, v2 - v0)
        ln = np.linalg.norm(fn)
        if ln > 1e-14:
            fn = fn / ln
        acc[i0] += fn
        acc[i1] += fn
        acc[i2] += fn
    norms = np.linalg.norm(acc, axis=1, keepdims=True)
    norms = np.where(norms < 1e-14, 1.0, norms)
    acc /= norms
    return acc.astype(np.float32)


NAN_TEXCOORD: float = -1.0
"""Sentinel texcoord for NaN/no-data vertices.

The viewer binds the 1D colormap texture with ``GL_CLAMP_TO_BORDER`` and a dark
border color, so any vertex whose texcoord is < 0 (or > 1) samples the border
and is rendered "color-less" rather than getting an arbitrary mid-LUT color.
"""


def scalars_to_texcoord_u(
    scalars: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    color_mode: str,
    piece_knots: list[float],
    n_bins: int,
) -> np.ndarray:
    """Per-vertex texture coordinate for the 1D LUT.

    Finite values are mapped into ``[0, 1]``. Non-finite values (NaN/inf) get
    :data:`NAN_TEXCOORD` so the fragment samples the texture's border color
    instead of producing a misleading mid-LUT color.

    For ``standard`` mode, ``n_bins`` does **not** quantize each vertex: bin
    discretisation is applied only when building the LUT texture
    (:func:`build_1d_lut_rgb`), so rasterisation interpolates smooth ``u`` like
    ParaView's surface mapper.
    """
    v = np.asarray(scalars, dtype=np.float64).reshape(-1)
    finite = np.isfinite(v)
    if vmax - vmin < 1e-12:
        t = np.zeros_like(v)
    else:
        t = (v - vmin) / (vmax - vmin)
    t_clipped = np.clip(t, 0.0, 1.0)

    if color_mode == "custom":
        return np.where(finite, t_clipped, NAN_TEXCOORD).astype(np.float32)

    u = t_to_cmap_u(t_clipped, piece_knots)
    # Do **not** quantize ``u`` per vertex here. Quantization belongs in the 1D
    # LUT texture (see :func:`build_1d_lut_rgb`): vertices keep full-precision
    # cmap coordinates so GL linearly interpolates ``u`` across each triangle
    # before ``texture1D``, giving smooth iso-color boundaries like ParaView.
    # Snapping corners to bin centres first (old behaviour) aligns band edges
    # with triangle edges and reads as a zigzag on coarse meshes.
    return np.where(finite, u, NAN_TEXCOORD).astype(np.float32)


def fill_nan_with_closest_neighbour(
    scalars: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN per-vertex scalars from the closest non-NaN triangle co-vertex.

    Rules (per triangle, applied globally via per-vertex propagation):

      * one NaN vertex in a triangle  -> NaN gets the closer of the two
        non-NaN co-vertices in 3D Euclidean distance.
      * two NaN vertices in a triangle -> both NaN verts inherit the single
        non-NaN co-vertex's value (the triangle ends up uniform).
      * three NaN vertices            -> stays NaN, and the triangle is
        returned in ``drop_tri_mask`` so the renderer can omit it from the
        index buffer (no smeared fill across an empty patch boundary).

    The propagation is done once per vertex by scanning all incident triangles
    and picking the globally closest non-NaN co-vertex; this matches the
    user-requested behaviour without forcing a costly un-shared mesh upload.

    Returns
    -------
    filled : ``(n_verts,) float64`` — original scalars with NaNs replaced
        wherever at least one incident triangle had a non-NaN co-vertex.
    drop_tri_mask : ``(n_tris,) bool`` — ``True`` for triangles whose all
        three vertices remained NaN after propagation (callers should skip
        these in the index buffer).
    """
    s = np.asarray(scalars, dtype=np.float64).reshape(-1)
    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    n = s.size
    filled = s.copy()
    nan_mask = ~np.isfinite(s)
    if not nan_mask.any() or F.size == 0:
        # Either nothing to fill, or no triangles to drop.
        all_nan_tri = np.zeros(F.shape[0], dtype=bool) if F.size else np.zeros(0, dtype=bool)
        return filled, all_nan_tri

    # For each NaN vertex, find the closest non-NaN co-vertex across all
    # triangles that include it. Track the running minimum distance.
    best_dist = np.full(n, np.inf, dtype=np.float64)
    best_val = np.full(n, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(s)

    for col in (0, 1, 2):
        v_self = F[:, col]
        # Two co-vertex columns
        c1, c2 = (col + 1) % 3, (col + 2) % 3
        for other_col in (c1, c2):
            v_other = F[:, other_col]
            # Only triangles where self is NaN and other is finite
            mask = nan_mask[v_self] & finite_mask[v_other]
            if not mask.any():
                continue
            vs = v_self[mask]
            vo = v_other[mask]
            d = np.linalg.norm(V[vs] - V[vo], axis=1)
            # For each NaN vertex, keep the minimum distance candidate.
            # Use np.minimum.at-style reduction via sort + groupby (vectorised).
            order = np.argsort(vs, kind="stable")
            vs_s = vs[order]
            d_s = d[order]
            val_s = s[vo][order]
            # boundary detection: where the vertex id changes
            change = np.concatenate(([True], vs_s[1:] != vs_s[:-1]))
            seg_starts = np.where(change)[0]
            seg_ends = np.concatenate([seg_starts[1:], [vs_s.size]])
            for start, end in zip(seg_starts, seg_ends):
                v = int(vs_s[start])
                local_min = d_s[start:end].argmin()
                cand_d = float(d_s[start + local_min])
                cand_v = float(val_s[start + local_min])
                if cand_d < best_dist[v]:
                    best_dist[v] = cand_d
                    best_val[v] = cand_v

    fill_mask = nan_mask & np.isfinite(best_val)
    if fill_mask.any():
        filled[fill_mask] = best_val[fill_mask]

    # Triangles all of whose vertices stayed NaN after propagation: drop them.
    still_nan = ~np.isfinite(filled)
    drop_tri_mask = still_nan[F[:, 0]] & still_nan[F[:, 1]] & still_nan[F[:, 2]]
    return filled, drop_tri_mask

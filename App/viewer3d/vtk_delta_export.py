"""Write per-delta-metric surface fields as legacy ASCII ``.vtk`` polydata.

Uses the same bounded harmonic pipeline as the 3D viewer (FMM / heat / Dijkstra
geodesic cut + local harmonic fill). Files are written with a VTK **legacy**
header (``# vtk DataFile Version 4.1``) and classic ``POLYGONS`` layout so
ParaView and older VTK readers open them reliably — no ``vtk`` Python package
is required for export.

Each export follows the Carto legacy layout: ``POINT_DATA`` with packed
``double`` scalars in ``[0, 1]``, ``LOOKUP_TABLE lookup_table`` (1000 rows,
four color bands), then ``NORMALS Normals float``.
"""

from __future__ import annotations

import math
import re
import traceback
from pathlib import Path

import numpy as np

from . import colormap as cm
from . import laplacian as lap

VTK_LEGACY_DATAFILE_VERSION = "4.1"
VTK_LUT_NAME = "lookup_table"
VTK_LUT_SIZE = 1000
VTK_SCALAR_NAME = "scalars"


def _safe_array_name(tag: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(tag).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "scalar"


def _safe_filename(tag: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(tag).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "delta")[:120]


def interpolate_delta_on_mesh(
    verts_raw: np.ndarray,
    tris: np.ndarray,
    values: dict[int, float],
    elec_raw: np.ndarray,
    elec_global_idx: list[int],
    *,
    radius_eff: float | None,
    global_pass: bool = False,
) -> np.ndarray | None:
    """Return per-vertex scalar array (``nan`` outside cut), or ``None`` on failure."""
    verts_raw = np.asarray(verts_raw, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    n = int(verts_raw.shape[0])
    if n == 0 or tris.size == 0:
        return None
    try:
        L = lap.cot_laplacian(verts_raw, tris)
    except Exception:
        traceback.print_exc()
        return None

    if elec_raw is None or getattr(elec_raw, "size", 0) == 0:
        return np.full(n, np.nan, dtype=np.float64)

    try:
        anc_idx_all = lap.map_points_to_vertices(verts_raw, np.asarray(elec_raw, dtype=np.float64))
    except Exception:
        traceback.print_exc()
        return None

    active_local: list[int] = []
    anc_val_list: list[float] = []
    for k, gidx in enumerate(elec_global_idx):
        if k >= int(anc_idx_all.size):
            break
        v = values.get(int(gidx))
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
        return np.full(n, np.nan, dtype=np.float64)

    active_local_arr = np.asarray(active_local, dtype=np.int64)
    anc_idx = anc_idx_all[active_local_arr]
    anc_val = np.asarray(anc_val_list, dtype=np.float64)

    mesh_graph = None
    try:
        mesh_graph = lap.build_mesh_graph(verts_raw, tris)
    except Exception:
        traceback.print_exc()

    dist = None
    r_eff = float(radius_eff) if radius_eff is not None and np.isfinite(radius_eff) and float(radius_eff) > 0 else None

    if r_eff is not None:
        if mesh_graph is not None:
            try:
                dist = lap.dijkstra_distances_from_anchors(mesh_graph, anc_idx_all, None)
            except Exception:
                traceback.print_exc()
                dist = None
        if dist is None:
            try:
                dist = lap.fmm_distances_from_anchors(verts_raw, tris, anc_idx_all, None)
            except Exception:
                traceback.print_exc()
                dist = None
        if dist is None:
            try:
                dist = lap.heat_geodesic_distances_from_anchors(
                    verts_raw, tris, anc_idx_all, radius=None, tau_reference_radius=r_eff
                )
            except Exception:
                traceback.print_exc()
                dist = None

    solve_mode = "global" if global_pass else "local"
    try:
        if r_eff is None:
            f = lap.harmonic_interpolate(L, anc_idx, anc_val)
        elif dist is not None:
            mask = np.zeros(dist.shape[0], dtype=bool)
            mask[active_local_arr] = True
            f = lap.harmonic_interpolate_bounded_cached(
                L, dist, anc_idx, anc_val, mask, float(r_eff), mode=solve_mode
            )
        elif mesh_graph is not None:
            f = lap.harmonic_interpolate_bounded(
                L, mesh_graph, anc_idx, anc_val, float(r_eff), mode=solve_mode
            )
        else:
            f = lap.harmonic_interpolate(L, anc_idx, anc_val)
    except Exception:
        traceback.print_exc()
        return None
    return np.asarray(f, dtype=np.float64)


def _normalize_scalars_01(arr: np.ndarray) -> np.ndarray:
    """Map finite values to ``[0, 1]``; non-finite → ``0``."""
    v = np.asarray(arr, dtype=np.float64).reshape(-1).copy()
    fin = np.isfinite(v)
    if not fin.any():
        return np.zeros_like(v)
    vmin = float(np.min(v[fin]))
    vmax = float(np.max(v[fin]))
    out = np.zeros_like(v)
    if vmax - vmin < 1e-12:
        out[fin] = 0.0
        return np.clip(out, 0.0, 1.0)
    out[fin] = (v[fin] - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


# Four-band Carto-style LUT (low → high scalar index). Alpha always 1.
_CARTO_LUT_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 0.0, 1.0, 1.0),   # blue
    (0.0, 1.0, 1.0, 1.0),   # cyan
    (1.0, 1.0, 0.0, 1.0),   # yellow
    (1.0, 0.0, 0.0, 1.0),   # red
)


def carto_lookup_table_rgba(*, lut_size: int = VTK_LUT_SIZE) -> np.ndarray:
    """Carto legacy LUT: ``lut_size`` rows, four equal color bands (low → high)."""
    n = int(lut_size)
    n_colors = len(_CARTO_LUT_COLORS)
    band = max(1, n // n_colors)
    rows: list[np.ndarray] = []
    for i, rgba in enumerate(_CARTO_LUT_COLORS):
        count = band if i < n_colors - 1 else max(0, n - band * (n_colors - 1))
        if count <= 0:
            continue
        rows.append(np.tile(np.asarray(rgba, dtype=np.float64), (count, 1)))
    out = np.vstack(rows) if rows else np.zeros((0, 4), dtype=np.float64)
    if out.shape[0] < n:
        pad = np.tile(np.asarray(_CARTO_LUT_COLORS[-1], dtype=np.float64), (n - out.shape[0], 1))
        out = np.vstack([out, pad])
    elif out.shape[0] > n:
        out = out[:n]
    return out


def build_vtk_lookup_table_rgba(
    *,
    cmap_name: str = "turbo",
    reverse_cmap: bool = False,
    n_bins: int = 256,
    color_mode: str = "standard",
    piece_knots: list[float] | None = None,
    custom_bins: list[dict] | None = None,
    lut_size: int = VTK_LUT_SIZE,
) -> np.ndarray:
    """Return ``(lut_size, 4)`` float64 RGBA table in ``[0, 1]`` for legacy VTK."""
    rgb_u8, _ = cm.build_1d_lut_rgb(
        lut_width=max(16, int(lut_size)),
        color_mode=str(color_mode or "standard"),
        cmap_name=str(cmap_name or "turbo"),
        reverse_cmap=bool(reverse_cmap),
        n_bins=max(2, int(n_bins)),
        piece_knots=list(piece_knots or []),
        custom_bins=list(custom_bins or []),
        vmin=0.0,
        vmax=1.0,
    )
    rgb = np.asarray(rgb_u8, dtype=np.float64).reshape(-1, 3) / 255.0
    if rgb.shape[0] != int(lut_size):
        idx = np.linspace(0, max(0, rgb.shape[0] - 1), int(lut_size))
        rgb = rgb[np.round(idx).astype(np.int64)]
    alpha = np.ones((rgb.shape[0], 1), dtype=np.float64)
    return np.concatenate([rgb, alpha], axis=1)


def _packed_double_lines(arr_flat: np.ndarray, *, per_line: int = 9) -> list[str]:
    """Packed ASCII doubles (Carto-style, high precision)."""
    a = np.asarray(arr_flat, dtype=np.float64).ravel()
    if a.size == 0:
        return []
    strs = [f"{v:.10g}" for v in a]
    lines: list[str] = []
    for start in range(0, int(a.size), per_line):
        lines.append(" ".join(strs[start : start + per_line]) + " ")
    return lines


def swap_triangle_winding(tris: np.ndarray) -> np.ndarray:
    """Reverse each triangle (i0,i1,i2) → (i0,i2,i1) for opposite face normal."""
    t = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    return t[:, [0, 2, 1]].copy()


def compute_carto_vtk_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Outward unit vertex normals for Carto legacy VTK (area-weighted, lit surfaces)."""
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    n_v = int(verts.shape[0])
    acc = np.zeros((n_v, 3), dtype=np.float64)
    for tri in tris:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        fn = np.cross(v1 - v0, v2 - v0)
        acc[i0] += fn
        acc[i1] += fn
        acc[i2] += fn

    center = verts.mean(axis=0)
    radial = verts - center
    out = np.zeros((n_v, 3), dtype=np.float64)
    for i in range(n_v):
        n = acc[i]
        ln = float(np.linalg.norm(n))
        r = radial[i]
        rn = float(np.linalg.norm(r))
        if ln < 1e-14:
            n = r / rn if rn > 1e-14 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            n = n / ln
        if rn > 1e-14 and float(np.dot(n, r)) < 0.0:
            n = -n
        out[i] = n
    return out


def _packed_float_lines(arr_flat: np.ndarray, *, per_line: int = 9, fmt: str = "%g") -> list[str]:
    """Format a flat float array as text lines packed ``per_line`` values per line.

    Each returned line ends with the values separated by single spaces
    plus a trailing space (no line terminator) — the writer appends the
    chosen EOL. This matches the packed Carto-style ``.vtk`` layout
    (e.g. ``-2.36088 -1380.46 170.028 ... -3.8013 -1379.97 170.028 ``).
    """
    a = np.asarray(arr_flat).ravel()
    n = int(a.size)
    if n == 0:
        return []
    strs = [fmt % v for v in a]
    lines: list[str] = []
    for start in range(0, n, per_line):
        lines.append(" ".join(strs[start : start + per_line]) + " ")
    return lines


def write_vtk_polydata(
    path: str | Path,
    verts: np.ndarray,
    tris: np.ndarray,
    point_arrays: dict[str, np.ndarray],
    *,
    patient_name: str | None = None,
    lookup_table_rgba: np.ndarray | None = None,
    include_normals: bool = True,
    swap_winding: bool = True,
) -> None:
    """Legacy ASCII ``.vtk`` polydata matching Carto export layout."""
    path = Path(path)
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if swap_winding:
        tris = swap_triangle_winding(tris)
    n = int(verts.shape[0])
    ntri = int(tris.shape[0])
    if not point_arrays:
        raise ValueError("point_arrays must contain at least one array")
    if len(point_arrays) != 1:
        raise ValueError("write_vtk_polydata expects exactly one scalar array per file")

    _name, arr = next(iter(point_arrays.items()))
    if np.asarray(arr, dtype=np.float64).reshape(-1).size != n:
        raise ValueError(f"array {_name!r} length != nverts {n}")

    scalars_01 = _normalize_scalars_01(arr)

    lut = lookup_table_rgba
    if lut is None:
        lut = carto_lookup_table_rgba()
    lut = np.asarray(lut, dtype=np.float64).reshape(-1, 4)
    if lut.shape[0] != VTK_LUT_SIZE:
        raise ValueError(f"lookup table must have {VTK_LUT_SIZE} entries, got {lut.shape[0]}")

    pname = str(patient_name or "").strip()
    desc = f"PatientData {pname}" if pname else "PatientData"
    if len(desc) > 256:
        desc = desc[:256]

    eol = "\r\n"
    poly_list_len = ntri * 4
    with path.open("w", newline="", encoding="ascii", errors="replace") as fp:
        fp.write(f"# vtk DataFile Version {VTK_LEGACY_DATAFILE_VERSION}{eol}")
        fp.write(f"{desc}{eol}")
        fp.write(f"ASCII{eol}")
        fp.write(f"DATASET POLYDATA{eol}")
        fp.write(f"POINTS {n} float{eol}")
        for line in _packed_float_lines(verts, per_line=9, fmt="%g"):
            fp.write(line + eol)
        fp.write(eol)
        fp.write(f"POLYGONS {ntri} {poly_list_len}{eol}")
        fp.writelines(f"3 {ia} {ib} {ic} {eol}" for ia, ib, ic in tris.tolist())
        fp.write(eol)
        fp.write(f"POINT_DATA {n}{eol}")
        fp.write(f"SCALARS {VTK_SCALAR_NAME} double{eol}")
        fp.write(f"LOOKUP_TABLE {VTK_LUT_NAME}{eol}")
        for line in _packed_double_lines(scalars_01, per_line=9):
            fp.write(line + eol)
        fp.write(f"LOOKUP_TABLE {VTK_LUT_NAME} {VTK_LUT_SIZE}{eol}")
        for row in lut:
            fp.write(
                f"{float(row[0]):g} {float(row[1]):g} {float(row[2]):g} {float(row[3]):g}{eol}"
            )
        if include_normals:
            normals = compute_carto_vtk_normals(verts, tris)
            fp.write(eol)
            fp.write(f"NORMALS Normals float{eol}")
            for line in _packed_float_lines(normals.ravel(), per_line=9, fmt="%g"):
                fp.write(line + eol)


def export_all_delta_metrics(
    out_dir: str | Path,
    *,
    carto: object,
    provider: object,
    elec_raw: np.ndarray,
    elec_global_idx: list[int],
    interpolation_radius: float | None,
    default_radius_fn,
    global_pass: bool = False,
    patient_name: str | None = None,
    lookup_table_rgba: np.ndarray | None = None,
    include_normals: bool = True,
    swap_winding: bool = True,
) -> int:
    """Write one ``.vtk`` per delta metric key. Returns number of files written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verts = np.asarray(getattr(carto, "vertices", None), dtype=np.float64)
    tris = np.asarray(getattr(carto, "triangles", None), dtype=np.int64)
    if verts.size == 0 or tris.size == 0:
        raise RuntimeError("Mesh vertices/triangles are not loaded on carto object.")

    keys = list(provider.get_delta_metric_keys() or [])
    if not keys:
        return 0

    rad = interpolation_radius
    if rad is None or not (math.isfinite(float(rad)) and float(rad) > 0):
        rad = float(default_radius_fn()) if callable(default_radius_fn) else None

    lut = lookup_table_rgba
    if lut is None:
        lut = carto_lookup_table_rgba()

    n_written = 0
    for key in keys:
        try:
            vals = provider.get_delta_values_for(key) or {}
            field = interpolate_delta_on_mesh(
                verts,
                tris,
                vals,
                elec_raw,
                elec_global_idx,
                radius_eff=rad,
                global_pass=global_pass,
            )
            if field is None:
                continue
            fname = _safe_filename(key) + ".vtk"
            write_vtk_polydata(
                out_dir / fname,
                verts,
                tris,
                {key: field},
                patient_name=patient_name,
                lookup_table_rgba=lut,
                include_normals=include_normals,
                swap_winding=swap_winding,
            )
            n_written += 1
        except Exception:
            traceback.print_exc()
    return n_written

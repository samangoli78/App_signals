"""Write per-delta-metric surface fields as legacy ASCII ``.vtk`` polydata.

Uses the same bounded harmonic pipeline as the 3D viewer (FMM / heat / Dijkstra
geodesic cut + local harmonic fill). Files are written with a VTK **legacy**
header (``# vtk DataFile Version 4.1``) and classic ``POLYGONS`` layout so
ParaView and older VTK readers open them reliably — no ``vtk`` Python package
is required for export.

Non-finite mesh values are replaced by :data:`VTK_EXPORT_NO_DATA` (documented
in the file's description line).
"""

from __future__ import annotations

import math
import re
import traceback
from pathlib import Path

import numpy as np

from . import laplacian as lap

VTK_LEGACY_DATAFILE_VERSION = "4.1"
VTK_EXPORT_NO_DATA = -9999.0


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
    nodata: float = VTK_EXPORT_NO_DATA,
) -> None:
    """Legacy ASCII ``.vtk`` polydata in the Carto/3DS packed style.

    Layout (CRLF line endings, trailing space on every packed-data line,
    matches ``model/Aorta.vtk``)::

        # vtk DataFile Version 4.1
        <description>

        ASCII
        DATASET POLYDATA
        POINTS n float
        x y z x y z x y z
        ...
        POLYGONS m 4m
        3 i j k
        ...

        POINT_DATA n
        SCALARS <name> float 1
        LOOKUP_TABLE default
        v v v v v v v v v
        ...

    Vectors and scalars are written packed nine values per line at the
    default ``%g`` precision (six significant digits).
    """
    path = Path(path)
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    n = int(verts.shape[0])
    ntri = int(tris.shape[0])
    if not point_arrays:
        raise ValueError("point_arrays must contain at least one array")
    if len(point_arrays) != 1:
        raise ValueError("write_vtk_polydata expects exactly one scalar array per file")

    name, arr = next(iter(point_arrays.items()))
    a = np.asarray(arr, dtype=np.float64).reshape(-1).copy()
    if a.size != n:
        raise ValueError(f"array {name!r} length {a.size} != nverts {n}")
    bad = ~np.isfinite(a)
    a[bad] = float(nodata)

    sname = _safe_array_name(name)
    desc = f"{sname} delta surface export NO_DATA={float(nodata):g}"
    if len(desc) > 256:
        desc = desc[:256]

    eol = "\r\n"
    poly_list_len = ntri * 4
    with path.open("w", newline="", encoding="ascii", errors="replace") as fp:
        fp.write(f"# vtk DataFile Version {VTK_LEGACY_DATAFILE_VERSION}{eol}")
        fp.write(f"{desc}{eol}")
        fp.write(eol)  # blank line between title and ASCII (matches example)
        fp.write(f"ASCII{eol}")
        fp.write(f"DATASET POLYDATA{eol}")
        fp.write(f"POINTS {n} float{eol}")
        for line in _packed_float_lines(verts, per_line=9, fmt="%g"):
            fp.write(line + eol)
        fp.write(f"POLYGONS {ntri} {poly_list_len}{eol}")
        # One polygon per line: ``3 v0 v1 v2 `` + EOL.
        fp.writelines(f"3 {ia} {ib} {ic} {eol}" for ia, ib, ic in tris.tolist())
        fp.write(eol)  # blank line between POLYGONS and POINT_DATA
        fp.write(f"POINT_DATA {n}{eol}")
        fp.write(f"SCALARS {sname} float 1{eol}")
        fp.write(f"LOOKUP_TABLE default{eol}")
        for line in _packed_float_lines(a, per_line=9, fmt="%g"):
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
            )
            n_written += 1
        except Exception:
            traceback.print_exc()
    return n_written

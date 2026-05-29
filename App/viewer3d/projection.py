"""Project arbitrary 3D points onto the closest point of a triangular mesh.

Prefers VTK's ``vtkCellLocator`` when available (already a dependency of the
parser) for fast, robust queries. Falls back to a vectorized numpy
"closest point on triangle" routine (Ericson's Voronoi-region algorithm)
otherwise.
"""

from __future__ import annotations

import traceback

import numpy as np


def project_points_to_mesh(
    verts: np.ndarray,
    tris: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Return ``(N, 3)`` array of points projected onto the mesh surface.

    Each ``points[i]`` is mapped to its closest point on any triangle of the
    mesh given by ``verts`` (V, 3) and ``tris`` (T, 3).
    """
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if verts.size == 0 or tris.size == 0 or points.size == 0:
        return points.copy()

    try:
        return _project_with_vtk(verts, tris, points)
    except Exception:
        traceback.print_exc()
    return _project_with_numpy(verts, tris, points)


def _project_with_vtk(verts: np.ndarray, tris: np.ndarray, points: np.ndarray) -> np.ndarray:
    import vtk  # local import: heavy module, fail soft to numpy fallback

    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(verts.shape[0])
    for i, v in enumerate(verts):
        vtk_points.SetPoint(i, float(v[0]), float(v[1]), float(v[2]))

    polys = vtk.vtkCellArray()
    for tri in tris:
        polys.InsertNextCell(3)
        polys.InsertCellPoint(int(tri[0]))
        polys.InsertCellPoint(int(tri[1]))
        polys.InsertCellPoint(int(tri[2]))

    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_points)
    pd.SetPolys(polys)

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(pd)
    locator.BuildLocator()

    out = np.empty_like(points)
    closest = [0.0, 0.0, 0.0]
    cellId = vtk.mutable(0)
    subId = vtk.mutable(0)
    dist2 = vtk.mutable(0.0)
    for i, p in enumerate(points):
        locator.FindClosestPoint(
            [float(p[0]), float(p[1]), float(p[2])],
            closest,
            cellId,
            subId,
            dist2,
        )
        out[i, 0] = float(closest[0])
        out[i, 1] = float(closest[1])
        out[i, 2] = float(closest[2])
    return out


def _project_with_numpy(verts: np.ndarray, tris: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Pure-numpy closest-point-on-mesh.

    Per point: vectorized closest-point-on-triangle (Ericson, "Real-Time
    Collision Detection") evaluated across every triangle, then argmin.
    """
    A = verts[tris[:, 0]]
    B = verts[tris[:, 1]]
    C = verts[tris[:, 2]]
    AB = B - A
    AC = C - A
    out = np.empty_like(points)
    for i, p in enumerate(points):
        AP = p - A
        d1 = np.einsum("ij,ij->i", AB, AP)
        d2 = np.einsum("ij,ij->i", AC, AP)

        BP = p - B
        d3 = np.einsum("ij,ij->i", AB, BP)
        d4 = np.einsum("ij,ij->i", AC, BP)

        CP = p - C
        d5 = np.einsum("ij,ij->i", AB, CP)
        d6 = np.einsum("ij,ij->i", AC, CP)

        Q = A.copy()

        mA = (d1 <= 0) & (d2 <= 0)

        mB = (~mA) & (d3 >= 0) & (d4 <= d3)
        Q[mB] = B[mB]

        vc = d1 * d4 - d3 * d2
        mAB = (~mA) & (~mB) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        denom = d1 - d3
        denom_safe = np.where(denom != 0, denom, 1.0)
        v = d1 / denom_safe
        Q[mAB] = A[mAB] + v[mAB, None] * AB[mAB]

        mC = (~(mA | mB | mAB)) & (d6 >= 0) & (d5 <= d6)
        Q[mC] = C[mC]

        vb = d5 * d2 - d1 * d6
        mAC = (~(mA | mB | mAB | mC)) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        denom_ac = d2 - d6
        denom_ac_safe = np.where(denom_ac != 0, denom_ac, 1.0)
        w = d2 / denom_ac_safe
        Q[mAC] = A[mAC] + w[mAC, None] * AC[mAC]

        va = d3 * d6 - d5 * d4
        mBC = (~(mA | mB | mAB | mC | mAC)) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        denom_bc = (d4 - d3) + (d5 - d6)
        denom_bc_safe = np.where(denom_bc != 0, denom_bc, 1.0)
        w2 = (d4 - d3) / denom_bc_safe
        Q[mBC] = B[mBC] + w2[mBC, None] * (C[mBC] - B[mBC])

        mInside = ~(mA | mB | mAB | mC | mAC | mBC)
        denom_i = va + vb + vc
        denom_i_safe = np.where(denom_i != 0, denom_i, 1.0)
        vv = vb / denom_i_safe
        ww = vc / denom_i_safe
        Q[mInside] = A[mInside] + AB[mInside] * vv[mInside, None] + AC[mInside] * ww[mInside, None]

        dist2 = np.einsum("ij,ij->i", Q - p, Q - p)
        fid = int(np.argmin(dist2))
        out[i] = Q[fid]
    return out

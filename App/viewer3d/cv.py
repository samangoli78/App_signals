"""Conduction velocity from a per-vertex local-activation-time field.

Standard formula on a triangulated surface (e.g. Bayly et al. 1998,
Cantwell et al. 2015):

1. On each triangle compute the linear gradient of LAT, ``g_f = ∇LAT``
   (a 3D vector in the triangle plane, units ms/mm).
2. Average the face gradients to vertices, weighted by 1/3 of the face
   area (the standard P1-FEM lumped-mass projection).
3. The conduction velocity at vertex ``v`` is

       CV(v) = 1 / |g_v|     (mm/ms)

4. The result is clipped at ``max_cv`` (1 mm/ms by default) so flat
   regions don't blow up to infinity, and vertices with no usable
   incident face (e.g. all-NaN LAT in their 1-ring) stay NaN.
"""

from __future__ import annotations

import numpy as np


def per_face_gradient(V: np.ndarray, F: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-triangle gradient vector of a vertex scalar field.

    Returns ``(n_faces, 3)`` vectors lying in each triangle plane and
    pointing toward increasing ``t``. Degenerate or non-finite faces
    return NaN.

    Linear-FEM identity: for a triangle with vertices ``p_i`` and
    values ``t_i``, with edge vectors ``e_i = p_{i+1} - p_{i-1}`` taken
    opposite to vertex ``i`` (indices mod 3) and face normal
    ``N = (p1 - p0) × (p2 - p0)``,

        ∇t = N × (Σ_i t_i e_i) / |N|^2.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64).ravel()
    if F.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]

    e0 = p2 - p1
    e1 = p0 - p2
    e2 = p1 - p0

    N = np.cross(p1 - p0, p2 - p0)
    N_sq = np.einsum("ij,ij->i", N, N)

    t0 = t[F[:, 0]]
    t1 = t[F[:, 1]]
    t2 = t[F[:, 2]]
    s = t0[:, None] * e0 + t1[:, None] * e1 + t2[:, None] * e2

    with np.errstate(divide="ignore", invalid="ignore"):
        grad = np.cross(N, s) / np.where(N_sq > 1e-24, N_sq, np.nan)[:, None]

    bad = ~(np.isfinite(t0) & np.isfinite(t1) & np.isfinite(t2))
    grad[bad] = np.nan
    return grad


def conduction_velocity_from_lat(
    V: np.ndarray,
    F: np.ndarray,
    lat_per_vertex: np.ndarray,
    *,
    max_cv: float = 1.0,
    min_grad: float = 1e-9,
) -> np.ndarray:
    """Per-vertex conduction velocity (mm/ms) from an interpolated LAT.

    ``V`` is in millimetres, ``lat_per_vertex`` in milliseconds. The
    result has the same length as ``V``; entries with no usable
    neighbourhood (all-NaN LAT, isolated vertex, ...) are NaN. Values
    are capped at ``max_cv`` so a numerically-flat region doesn't
    diverge.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    lat = np.asarray(lat_per_vertex, dtype=np.float64).ravel()

    n_v = V.shape[0]
    cv = np.full(n_v, np.nan, dtype=np.float64)
    if n_v == 0 or F.size == 0:
        return cv

    # 1) Per-face gradient of LAT.
    grad_face = per_face_gradient(V, F, lat)            # (n_f, 3) ms/mm

    # 2) Face areas (used as the lumped-mass weights, 1/3 per vertex).
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    # Drop faces whose gradient or area is non-finite (NaN LAT, sliver, ...).
    face_ok = np.all(np.isfinite(grad_face), axis=1) & np.isfinite(area) & (area > 0)
    weight = np.where(face_ok, area / 3.0, 0.0)         # (n_f,)
    g_w = np.where(face_ok[:, None], grad_face, 0.0)    # (n_f, 3)

    # 3) Scatter-add to vertices.
    sum_g = np.zeros((n_v, 3), dtype=np.float64)
    sum_w = np.zeros(n_v, dtype=np.float64)
    for k in range(3):
        np.add.at(sum_g, F[:, k], g_w * weight[:, None])
        np.add.at(sum_w, F[:, k], weight)

    safe = sum_w > 0
    grad_v = np.zeros((n_v, 3), dtype=np.float64)
    grad_v[safe] = sum_g[safe] / sum_w[safe][:, None]
    mag = np.linalg.norm(grad_v, axis=1)                # ms/mm at each vertex

    # 4) CV = 1 / |∇LAT|, capped. Plateaus (mag ~ 0) saturate at max_cv.
    has_grad = safe & np.isfinite(mag)
    mag_eff = np.where(has_grad & (mag > float(min_grad)), mag, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_raw = 1.0 / mag_eff
    # plateau -> NaN above -> treat as "saturated at max_cv"
    cv_raw = np.where(has_grad & ~np.isfinite(cv_raw), float(max_cv), cv_raw)
    cv[has_grad] = np.minimum(cv_raw[has_grad], float(max_cv))
    return cv

"""Small geometry utilities for the 3D viewer (icosphere + batched sphere copies)."""

from __future__ import annotations

import numpy as np


def icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices Nx3, triangles Mx3)`` of a unit icosahedron."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = np.array(
        [
            (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    tris = np.array(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int32,
    )
    return verts, tris


def subdivide(verts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cache: dict[tuple[int, int], int] = {}
    v_list = list(map(tuple, verts.tolist()))

    def mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in cache:
            return cache[key]
        pa = np.array(v_list[a])
        pb = np.array(v_list[b])
        pm = (pa + pb) * 0.5
        pm = pm / np.linalg.norm(pm)
        v_list.append(tuple(pm))
        cache[key] = len(v_list) - 1
        return cache[key]

    new_tris = []
    for a, b, c in tris:
        ab = mid(int(a), int(b))
        bc = mid(int(b), int(c))
        ca = mid(int(c), int(a))
        new_tris.append((int(a), ab, ca))
        new_tris.append((int(b), bc, ab))
        new_tris.append((int(c), ca, bc))
        new_tris.append((ab, bc, ca))
    return np.array(v_list, dtype=np.float64), np.array(new_tris, dtype=np.int32)


def icosphere(subdivisions: int = 1) -> tuple[np.ndarray, np.ndarray]:
    v, t = icosahedron()
    for _ in range(max(0, int(subdivisions))):
        v, t = subdivide(v, t)
    return v.astype(np.float32), t.astype(np.int32)


def build_sphere_batch(
    centers: np.ndarray,
    radius: float,
    base_verts: np.ndarray,
    base_tris: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(verts, normals, tris, vertex_sphere_idx)`` for N spheres.

    All inputs/outputs are float32 / int32 except ``vertex_sphere_idx`` which is
    int32. Sphere normals are radial (same as base sphere vertex coords because
    the base sphere is unit-radius).
    """
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    n_spheres = centers.shape[0]
    v_per = base_verts.shape[0]
    t_per = base_tris.shape[0]

    verts = np.tile(base_verts * float(radius), (n_spheres, 1))
    offsets = np.repeat(centers, v_per, axis=0)
    verts = (verts + offsets).astype(np.float32)
    normals = np.tile(base_verts.astype(np.float32), (n_spheres, 1))
    vertex_sphere_idx = np.repeat(np.arange(n_spheres, dtype=np.int32), v_per)

    tris = np.tile(base_tris, (n_spheres, 1))
    offsets_idx = np.repeat(np.arange(n_spheres, dtype=np.int32) * v_per, t_per).reshape(-1, 1)
    tris = (tris + offsets_idx).astype(np.int32)
    return verts, normals, tris, vertex_sphere_idx

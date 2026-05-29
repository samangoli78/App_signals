"""ID <-> RGB encoding for back-buffer color picking.

Convention
----------
- ID 0  : background / empty space.
- ID 1.. ``n_spheres`` : electrode spheres.
- ID ``n_spheres + 1`` .. ``n_spheres + n_triangles`` : mesh triangles.
"""

from __future__ import annotations

import numpy as np


def id_to_rgb(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint32).reshape(-1)
    r = (ids & 0xFF).astype(np.uint8)
    g = ((ids >> 8) & 0xFF).astype(np.uint8)
    b = ((ids >> 16) & 0xFF).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def rgb_to_id(rgb) -> int:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return r | (g << 8) | (b << 16)


def split_id(pick_id: int, n_spheres: int, n_triangles: int) -> tuple[str, int]:
    """Return ``(kind, local_index)``.

    ``kind`` is ``"empty"``, ``"sphere"`` or ``"triangle"``;
    ``local_index`` is the index within that pool (0-based).
    """
    if pick_id <= 0:
        return "empty", -1
    if pick_id <= n_spheres:
        return "sphere", pick_id - 1
    if pick_id <= n_spheres + n_triangles:
        return "triangle", pick_id - 1 - n_spheres
    return "empty", -1

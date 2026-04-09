import numpy as np
import math
from .Helper import clamp, look_at, perspective, normalize


# ----------------- mesh generation -----------------

def make_uv_sphere_exploded(lat=32, lon=48, radius=0.9, center=(0.0, 0.0)):
    cx, cz = float(center[0]), float(center[1])

    verts = []
    for i in range(lat + 1):
        th = math.pi * i / lat
        y = math.cos(th)
        r = math.sin(th)
        for j in range(lon + 1):
            ph = 2 * math.pi * j / lon

            # sphere point around origin (unit sphere)
            x0 = r * math.sin(ph)
            z0 = r * math.cos(ph)

            # scale first
            x = x0 * radius
            yv = y  * radius
            z = z0 * radius

            # then translate (DO NOT scale the translation)
            x += cx
            z += cz

            verts.append((x, yv, z))

    verts = np.array(verts, dtype=np.float32)

    def vid(i, j):
        return i * (lon + 1) + j

    tris = []
    for i in range(lat):
        for j in range(lon):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i + 1, j + 1)
            d = vid(i, j + 1)
            tris.append((a, b, c))
            tris.append((a, c, d))

    pos, nor, bary, face = [], [], [], []
    b0, b1, b2 = (1,0,0), (0,1,0), (0,0,1)

    center3 = np.array([cx, 0.0, cz], dtype=np.float32)

    face_id = 1
    for ia, ib, ic in tris:
        pa, pb, pc = verts[ia], verts[ib], verts[ic]

        # normals must be from the sphere center, not from origin
        na = normalize(pa - center3)
        nb = normalize(pb - center3)
        nc = normalize(pc - center3)

        pos.extend([pa, pb, pc])
        nor.extend([na, nb, nc])
        bary.extend([b0, b1, b2])
        face.extend([face_id, face_id, face_id])
        face_id += 1

    return (
        np.asarray(pos, dtype=np.float32),
        np.asarray(nor, dtype=np.float32),
        np.asarray(bary, dtype=np.float32),
        np.asarray(face, dtype=np.uint32),
        face_id - 1
    )

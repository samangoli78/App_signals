import numpy as np
import math
def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


# ----------------- mesh generation -----------------

def make_uv_sphere_exploded(lat=32, lon=48, radius=0.9,center=[0,0]):
    """
    Returns exploded triangle arrays:
      pos: (N,3) float32
      nor: (N,3) float32
      bary:(N,3) float32
      face:(N,)  uint32  (same face id repeated 3 times)
    """
    # create vertex grid
    verts = []
    for i in range(lat + 1):
        th = math.pi * i / lat
        y = math.cos(th)
        r = math.sin(th)
        for j in range(lon + 1):
            ph = 2 * math.pi * j / lon
            x = center[0]+r * math.sin(ph)
            z = center[1]+r * math.cos(ph)
            verts.append((x, y, z))
    verts = np.array(verts, dtype=np.float32) * radius

    def vid(i, j):
        return i * (lon + 1) + j

    # indexed triangles
    tris = []
    for i in range(lat):
        for j in range(lon):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i + 1, j + 1)
            d = vid(i, j + 1)
            # two triangles per quad (skip degenerate caps are fine here)
            tris.append((a, b, c))
            tris.append((a, c, d))

    # explode + attach bary + face id
    pos = []
    nor = []
    bary = []
    face = []

    b0 = (1.0, 0.0, 0.0)
    b1 = (0.0, 1.0, 0.0)
    b2 = (0.0, 0.0, 1.0)

    face_id = 1
    for (ia, ib, ic) in tris:
        pa, pb, pc = verts[ia], verts[ib], verts[ic]

        # normals: sphere normal = normalized position (radius cancels)
        na = normalize(pa.copy())
        nb = normalize(pb.copy())
        nc = normalize(pc.copy())

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

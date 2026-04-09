
import numpy as np
import math

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def look_at(eye, center, up):
    eye = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M

def perspective(fovy_deg, aspect, znear, zfar):
    fovy = math.radians(fovy_deg)
    f = 1.0 / math.tan(fovy / 2.0)

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (zfar + znear) / (znear - zfar)
    M[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    M[3, 2] = -1.0
    return M

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n
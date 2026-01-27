# renderer.py  (DROP-IN: no FBO, no picking, just draw to screen)
import numpy as np
from OpenGL import GL
from Shader import Shader


def _perspective(fov_y_deg, aspect, z_near, z_far):
    f = 1.0 / np.tan(np.deg2rad(fov_y_deg) * 0.5)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / max(aspect, 1e-9)
    M[1, 1] = f
    M[2, 2] = (z_far + z_near) / (z_near - z_far)
    M[2, 3] = (2 * z_far * z_near) / (z_near - z_far)
    M[3, 2] = -1.0
    return M


def _look_at_target(eye, target, up=(0, 0, 1)):
    eye = np.asarray(eye, np.float32)
    target = np.asarray(target, np.float32)
    up = np.asarray(up, np.float32)

    f = target - eye
    f /= (np.linalg.norm(f) + 1e-12)

    s = np.cross(f, up)
    s /= (np.linalg.norm(s) + 1e-12)

    u = np.cross(s, f)
    u /= (np.linalg.norm(u) + 1e-12)

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] =  np.dot(f, eye)
    return M




"""
f = [
    cos(pitch) * cos(yaw),  # X
    cos(pitch) * sin(yaw),  # Y
    sin(pitch)              # Z
]"""  

def _vec_yaw_pitch(yaw, pitch, eps=1e-2):
    # clip pitch to avoid singularity
    max_pitch = np.pi * 0.5 - eps
    pitch = np.clip(pitch, -max_pitch, max_pitch)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # yaw in X-Y plane, pitch on Z
    f = np.array([
        sy * cp,
        cy * cp,
        sp
    ], dtype=np.float32)

    n = np.linalg.norm(f) + 1e-12
    return f / n



class Camera:
    def __init__(self, target=None):
        self.yaw = 0.0
        self.pitch = 0.0
        self.distance = 27.0
        self.target = target if target is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.orbit_sens = 0.005
        self.pitch_eps = 1e-2
        self.max_pitch = np.pi * 0.5 - self.pitch_eps
        self._w = 1
        self._h = 1

    def apply_drag(self, dl):
        if not dl:
            return
        dx, dy = dl
        self.yaw += float(dx) * self.orbit_sens
        self.pitch += float(dy) * self.orbit_sens
        self.pitch = float(np.clip(self.pitch, -self.max_pitch, self.max_pitch))
    def zoom(self, wheel_delta, zoom_sens=0.001):
        # wheel_delta: +120/-120 typical on Windows
        d = float(wheel_delta) * zoom_sens
        self.distance *= (1.0 - d)   # multiplicative zoom feels nicer
        self.distance = float(np.clip(self.distance, 0.5, 50.0))


    def set_viewport(self, w, h):
        self._w = max(1, int(w))
        self._h = max(1, int(h))

    def view_proj(self):
        w, h = self._w, self._h
        aspect = (w / h) if h else 1.0
        P = _perspective(60.0, aspect, 0.1, 100.0)

        f = _vec_yaw_pitch(self.yaw, self.pitch)
        eye = self.target - f * float(self.distance)
        V = _look_at_target(eye, self.target)

        return V, P

# tk_gl_shader_pick_sphere.py
# pip install PyOpenGL PyOpenGL_accelerate pyopengltk numpy

import math
import tkinter as tk
import numpy as np
from pyopengltk import OpenGLFrame
from OpenGL import GL
from Geometry import make_uv_sphere_exploded,normalize

# ----------------- small math helpers (numpy) -----------------

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

# ----------------- shaders -----------------

from Shader import BEAUTY_VS, BEAUTY_FS, PICK_VS, PICK_FS
from GL_Objects import bind

# ----------------- main widget -----------------

class Three_D_Frame(OpenGLFrame):
    def initgl(self):
        # --- GL sanity print ---
        renderer = GL.glGetString(GL.GL_RENDERER)
        version = GL.glGetString(GL.GL_VERSION)
        print("GL_RENDERER:", renderer.decode() if renderer else None)
        print("GL_VERSION :", version.decode() if version else None)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glEnable(GL.GL_POLYGON_SMOOTH)
        GL.glHint(GL.GL_POLYGON_SMOOTH_HINT, GL.GL_NICEST)

        GL.glClearColor(0.08, 0.08, 0.10, 1.0)

        # Mesh
        obj1=bind(world=self,obj_id=1).bind()
        obj2=bind(world=self,obj_id=2).bind(center=[3,2])
        self.objs=[obj1,obj2]
        


        # Camera state
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.yaw = 0.0
        self.pitch = 0.0
        self.distance = 3.0
        self.light_world = np.array([2.0, 2.0, 4.0], dtype=np.float32)

        # Mouse
        self._dragging = False
        self._press_xy = (0, 0)
        self._last_xy = (0, 0)

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

        self.bind("<MouseWheel>", self._on_wheel)      # Windows/macOS
        self.bind("<Button-4>", self._on_wheel_linux)  # Linux
        self.bind("<Button-5>", self._on_wheel_linux)

        self.focus_set()

        # Continuous redraw (simple/reliable)
        self.after(16, self._tick)

    # ---- redraw scheduling ----
    def _tick(self):
        self._request_redraw()
        self.after(16, self._tick)

    def _request_redraw(self):
        # pyopengltk draws via tkExpose -> _display -> redraw
        try:
            self.tkExpose()
        except Exception:
            try:
                self._display()
            except Exception:
                self.update_idletasks()

    # ---- camera helpers ----
    def _eye_pos(self):
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        dirx = sy * cp
        diry = sp
        dirz = cy * cp
        return self.center + self.distance * np.array([dirx, diry, dirz], dtype=np.float32)

    def _matrices(self, w, h):
        eye = self._eye_pos()
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        view = look_at(eye, self.center, up)
        proj = perspective(45.0, w / float(h), 0.1, 200.0)
        model = np.eye(4, dtype=np.float32)
        mvp = proj @ view
        normal_mat = np.linalg.inv(model[:3, :3]).T.astype(np.float32)
        return eye, model, view, proj, mvp, normal_mat

    def _camera_basis(self):
        eye = self._eye_pos()
        f = normalize(self.center - eye)               # forward
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = normalize(np.cross(f, world_up))           # right
        u = np.cross(r, f)                             # up
        return r, u, f

    # ---- input ----
    def _on_press(self, e):
        self._dragging = True
        self._press_xy = (e.x, e.y)
        self._last_xy = (e.x, e.y)

    def _on_drag(self, e):
        if not self._dragging:
            return

        x0, y0 = self._last_xy
        dx = e.x - x0
        dy = e.y - y0
        self._last_xy = (e.x, e.y)

        shift = (e.state & 0x0001) != 0  # common shift bit in Tk

        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        min_dim = max(1, min(w, h))

        if shift:
            # pan in view plane
            pan_speed = 2.0 * self.distance / float(min_dim)
            r, u, _ = self._camera_basis()
            self.center = self.center + (-dx) * pan_speed * r + (dy) * pan_speed * u
        else:
            # orbit
            rot_speed = 2.5 / float(min_dim)
            self.yaw += dx * rot_speed
            self.pitch += dy * rot_speed
            self.pitch = clamp(self.pitch, -1.45, 1.45)

    def _on_release(self, e):
        self._dragging = False
        # click pick if it was a small movement (i.e., not a drag)
        px, py = self._press_xy
        dist2 = (e.x - px) ** 2 + (e.y - py) ** 2
        if dist2 <= 4:  # ~2px threshold
            self.pick_at(e.x, e.y)

    def _on_wheel(self, e):
        delta = e.delta / 120.0 if abs(e.delta) >= 120 else e.delta
        self._zoom(delta)

    def _on_wheel_linux(self, e):
        delta = 1.0 if e.num == 4 else -1.0
        self._zoom(delta)

    def _zoom(self, delta):
        self.distance *= math.exp(-0.15 * float(delta))
        #self.distance = clamp(self.distance, 0.3, 50.0)

    # ---- FBO management ----

    def redraw(self):
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())

        eye, model, view, proj, mvp, normal_mat = self._matrices(w, h)

        # (A) update pick buffers (you can keep this as-is)
        for obj in self.objs:
            obj._render_pick_pass(w, h, eye, model, mvp)

        # (B) now render beauty to screen: clear ONCE
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        for obj in self.objs:
            obj._render_beauty_pass(w, h, eye, model, mvp, normal_mat)

        GL.glFlush()


    # ---- picking ----
    def pick_at(self, x_tk, y_tk):
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())

        # Convert Tk coords (origin top-left) to GL coords (origin bottom-left)
        x = int(clamp(x_tk, 0, w - 1))
        y = int(clamp(h - 1 - y_tk, 0, h - 1))

        # Ensure pick pass is current for this frame
        eye, model, view, proj, mvp, normal_mat = self._matrices(w, h)
        selected=None
        for obj in self.objs:
            obj:bind
            obj._render_pick_pass(w, h, eye, model, mvp)
            result=obj.picked(x_tk,y_tk,x,y)
            if result is not None:
                selected=result
        print(f"[pick] at ({x_tk},{y_tk}) -> {selected}")

        


def main():
    root = tk.Tk()
    root.title("Tkinter OpenGL Shader Sphere + ID/Barycentric Picking")

    frame = Three_D_Frame(root, width=960, height=720)
    frame.pack(fill="both", expand=True)

    tk.Label(
        root,
        text="Left-drag: orbit | Shift+Left-drag: pan | Wheel: zoom | Click (no-drag): print object/face/bary",
        anchor="w"
    ).pack(fill="x")

    root.mainloop()

if __name__ == "__main__":
    main()
 
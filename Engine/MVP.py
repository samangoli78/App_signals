# tk_gl_shader_pick_sphere.py
# pip install PyOpenGL PyOpenGL_accelerate pyopengltk numpy

import math
import tkinter as tk
import numpy as np
from pyopengltk import OpenGLFrame
from OpenGL import GL
from .Geometry import make_uv_sphere_exploded
from typing import Optional,Tuple
import sys
import traceback

# ----------------- small math helpers (numpy) -----------------
from .Helper import clamp, look_at, perspective, normalize
from .gl_backend import fbo

# ----------------- shaders -----------------

from .Shader import BEAUTY_VS, BEAUTY_FS, PICK_VS, PICK_FS
from .GL_Objects import Object

# ----------------- main widget -----------------

class Three_D_Frame(OpenGLFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.objs = []
        # --- GL sanity print ---




        # Camera state
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.yaw = 0.0
        self.pitch = 0.0
        self.distance = 3.0
        self.eye_pos=self._eye_pos()
        self.light_world = np.array([20.0, 20.0, 40.0], dtype=np.float32)

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
        self.bind("<Motion>", lambda e:self.locate_from_mouse_position(e.x, e.y,self.winfo_width(),self.winfo_height()))
        

        self.pick_fbo = None
        self.pick_tex_obj = None
        self.pick_tex_face = None
        self.pick_tex_bary = None
        self.pick_depth = None
        self._fb_w = 0
        self._fb_h = 0

    def initgl(self):
        renderer = GL.glGetString(GL.GL_RENDERER)
        version = GL.glGetString(GL.GL_VERSION)
        print("GL_RENDERER:", renderer.decode() if renderer else None)
        print("GL_VERSION :", version.decode() if version else None)

        # initialize GL state
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glEnable(GL.GL_POLYGON_SMOOTH)
        GL.glHint(GL.GL_POLYGON_SMOOTH_HINT, GL.GL_NICEST)
        GL.glClearColor(0.08, 0.08, 0.10, 1.0)
        self.focus_set()

        # Continuous redraw (simple/reliable)
        self.after(16, self._tick)

    # $$$$$$$$$$$$$---- redraw scheduling main loop start ----$$$$$$$$$$$$$
    def _tick(self):
        self._request_redraw()
        self.after(16, self._tick)
        

    def _request_redraw(self):
        # pyopengltk draws via tkExpose -> _display -> redraw
  
        try:
            # it reads redraw make sure self has already implemented redraw method 
            self.sync_light() # sync light position to eye position before redraw
            self._display()
        except Exception as e:
            print("Error during redraw:", e)
            print(traceback.format_exc())
            self.update_idletasks()

    def sync_light(self):
        self.light_world=self.eye_pos
        




    #$$$$$$$$$$$$$$---- redraw scheduling main loop closed ----$$$$$$$$$$$$$

     # ---- redraw replace the th redraw ----

    def redraw(self):
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        # (A) update pick buffers (you can keep this as-is)
        #for obj in self.objs:
        #    obj._render_pick_pass(w, h, eye, model, mvp)

        # (B) now render beauty to screen: clear ONCE
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glViewport(0, 0, w, h)
        for obj in self.objs:
            obj._render_beauty_pass()

        GL.glFlush()
    
    def _create_update_pick_fbo(self, w, h):
        # 1) Already created and correct size -> keep it
        if self.pick_fbo is not None and self._fb_w == w and self._fb_h == h:
            return

        # 2) Created but wrong size -> delete
        if self.pick_fbo is not None:
            GL.glDeleteFramebuffers(1, [self.pick_fbo])
            GL.glDeleteTextures([self.pick_tex_obj, self.pick_tex_face, self.pick_tex_bary])
            GL.glDeleteRenderbuffers(1, [self.pick_depth])

            self.pick_fbo = None
            self.pick_tex_obj = self.pick_tex_face = self.pick_tex_bary = None
            self.pick_depth = None

        # 3) Create (either first time, or after resize)
        self._fb_w, self._fb_h = w, h
        self.pick_fbo, textures = fbo(w, h, layout="1u 1u 2f")
        self.pick_tex_obj, self.pick_tex_face, self.pick_tex_bary = textures

        # Depth attachment
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.pick_fbo)
        self.pick_depth = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.pick_depth)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, w, h)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                    GL.GL_RENDERBUFFER, self.pick_depth)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            raise RuntimeError(f"Pick FBO incomplete: 0x{status:X}")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    
    # ---- picking ----
    def pick_at(self, x_tk, y_tk):
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        self.locate_from_mouse_position(x_tk, y_tk,w,h)

    def locate_from_mouse_position(self, x_tk, y_tk, w, h):


        self._create_update_pick_fbo(w, h)
        x = int(clamp(x_tk, 0, w - 1))
        y = int(clamp(h - 1 - y_tk, 0, h - 1))

        # ========= 2-line live HUD (robust: save/restore cursor) =========
        # First call: print two lines once and SAVE the cursor at the top of them.
        if not hasattr(self, "_hud_saved"):
            sys.stdout.write("\n\n")          # reserve 2 lines
            sys.stdout.write("\x1b[2F")       # go to start of those 2 lines
            sys.stdout.write("\x1b[s")        # SAVE cursor position (top HUD line)
            sys.stdout.flush()
            self._hud_saved = True

        # Every call: RESTORE cursor to top HUD line, clear+rewrite 2 lines.
        sys.stdout.write("\x1b[u")            # RESTORE cursor (top HUD line)
        sys.stdout.write("\x1b[2K")           # clear line 1
        sys.stdout.write(f"[pick] mouse=({x_tk},{y_tk}) fb=({x},{y})\n")
        sys.stdout.write("\x1b[2K")           # clear line 2
        sys.stdout.write("[pick] ...\n")
        sys.stdout.flush()
        # ================================================================
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.pick_fbo)
        GL.glViewport(0, 0, w, h)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glClearBufferuiv(GL.GL_COLOR, 0, np.array([0], dtype=np.uint32))
        GL.glClearBufferuiv(GL.GL_COLOR, 1, np.array([0], dtype=np.uint32))
        GL.glClearBufferfv(GL.GL_COLOR, 2, np.array([0.0, 0.0], dtype=np.float32))
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        for obj in self.objs:
            obj._render_pick_pass()

        obj = np.zeros((1,), dtype=np.uint32)
        face = np.zeros((1,), dtype=np.uint32)
        bary_uv = np.zeros((2,), dtype=np.float32)

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glReadPixels(x, y, 1, 1, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, obj)

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        GL.glReadPixels(x, y, 1, 1, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, face)

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
        GL.glReadPixels(x, y, 1, 1, GL.GL_RG, GL.GL_FLOAT, bary_uv)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        obj_id = int(obj[0])
        if obj_id == 0:
            line2 = "[pick] none"
        else:
            face_id = int(face[0])
            b0 = float(bary_uv[0])
            b1 = float(bary_uv[1])
            b2 = 1.0 - b0 - b1
            line2 = f"[pick] obj={obj_id} face={face_id} bary=({b0:.3f},{b1:.3f},{b2:.3f})"

        # Update just the 2nd line (again: restore to top, then go down 1 line)
        sys.stdout.write("\x1b[u")            # top HUD line
        sys.stdout.write("\x1b[1E")           # move to next line (start of line)
        sys.stdout.write("\x1b[2K" + line2 + "\n")
        sys.stdout.flush()

        for obj in self.objs:
            obj:Object
            if obj.object_id == obj_id:
                self.center = obj.center.astype(np.float32)

                # sync orbit params to the new (eye_pos, center) so drag stays smooth
                v = self.eye_pos - self.center
                d = float(np.linalg.norm(v))
                if d > 1e-6:
                    self.distance = d
                    self.yaw = math.atan2(v[0], v[2])
                    self.pitch = math.asin(clamp(v[1] / d, -1.0, 1.0))

    # ------------------------------------------------------

        
  
    
    # ---- camera helpers ----

    def _camera_basis(self):
        eye = self.eye_pos
        f = normalize(self.center - eye)               # forward
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = normalize(np.cross(f, world_up))           # right
        u = np.cross(r, f)                             # up
        return r, u, f
    
    def _eye_pos(self):
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        dirx = sy * cp
        diry = sp
        dirz = cy * cp
        return self.center + self.distance * np.array([dirx, diry, dirz], dtype=np.float32)

    def _matrices(self, w, h,up:Optional[np.ndarray]=None,eye:Optional[np.ndarray]=None,center:Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :param self: the instance of the class
        :param w: width of the viewport
        :param h: height of the viewport
        :param up: up vector, default is (0,1,0)
        :param eye: eye position, default is from self._eye_pos()
        :param center: center position, default is from self.center
        :returns: eye position, model matrix, view matrix, 
        projection matrix, mvp matrix, normal matrix
        :rtype: tuple
        """
        if eye is None:
            eye = self.eye_pos
        if up is None:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if center is None:
            center = self.center

        view = look_at(eye, center, up)
        proj = perspective(45.0, w / float(h), 0.1, 2000.0)
        model = np.eye(4, dtype=np.float32)
        mvp = proj @ view
        normal_mat = np.linalg.inv(model[:3, :3]).T.astype(np.float32)
        return eye, model, view, proj, mvp, normal_mat

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

        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        min_dim = max(1, min(w, h))

        # rotation speed in radians per pixel-ish
        rot_speed = 2.5 / float(min_dim)
        yaw_delta   = dx * rot_speed
        pitch_delta = dy * rot_speed

        # vector from target to eye
        v = self.eye_pos - self.center
        d = np.linalg.norm(v)
        if d < 1e-6:
            return

        # current spherical angles from v (so we don't depend on stored yaw/pitch)
        # v = [x,y,z] where z forward in your convention
        yaw = math.atan2(v[0], v[2])
        pitch = math.asin(clamp(v[1] / d, -1.0, 1.0))

        # apply deltas
        yaw += yaw_delta
        pitch = clamp(pitch + pitch_delta, -1.45, 1.45)

        # rebuild v from yaw/pitch, keep same distance d
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        new_v = d * np.array([sy * cp, sp, cy * cp], dtype=np.float32)
        self.eye_pos = self.center + new_v

        # (optional) keep stored yaw/pitch in sync if you still want them elsewhere
        self.yaw = yaw
        self.pitch = pitch
        self.distance = d


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
        # delta > 0 : zoom in (closer), delta < 0 : zoom out (farther)
        v = self.eye_pos - self.center            # from target to eye
        d = np.linalg.norm(v)
        if d < 1e-6:
            return

        dir = v / d                               # unit direction (target -> eye)

        # exponential zoom factor (same feel as your old code)
        factor = math.exp(-0.15 * float(delta))   # delta>0 => factor<1 => zoom in
        new_d = d * factor

        # optional clamps to avoid going through/too far
        new_d = clamp(new_d, 0.05, 5000.0)

        self.eye_pos = self.center + dir * new_d


   


def main():
    root = tk.Tk()
    root.title("Tkinter OpenGL Shader Sphere + ID/Barycentric Picking")

    frame = Three_D_Frame(root, width=960, height=720)  # Manually call initgl to set up OpenGL state and objects
    frame.pack(fill="both", expand=True)


    tk.Label(
        root,
        text="Left-drag: orbit | Shift+Left-drag: pan | Wheel: zoom | Click (no-drag): print object/face/bary",
        anchor="w"
    ).pack(fill="x")
    def command():
        # Mesh
        obj1=Object(world=frame,obj_id=1).bind_sphere_default()
        obj2=Object(world=frame,obj_id=2).bind_sphere_default(center=[-3,2])
        frame.objs=[obj1,obj2]
    root.after(100, command)  # Schedule the command to run after 100ms (after initgl has been called)
    root.mainloop()

if __name__ == "__main__":
    main()


















from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from OpenGL import GL
from Geometry import make_uv_sphere_exploded
from Shader import BEAUTY_VS, BEAUTY_FS, PICK_VS, PICK_FS
from gl_backend import set_uniform_mat4, set_uniform_mat3, set_uniform_vec3, set_uniform_u1, link_program

if TYPE_CHECKING:
    from Shader import ShaderPickSphere



class bind:
    def __init__(self,world:ShaderPickSphere,obj_id):
                # Picking FBO (created on first draw / resize)
        self.world=world
        self.pick_fbo = None
        self.pick_tex_obj = None
        self.pick_tex_face = None
        self.pick_tex_bary = None
        self.pick_depth = None
        self._fb_w = 0
        self._fb_h = 0
        self.object_id=obj_id
    def bind(self,**kwargs):
        pos, nor, bary, face, num_faces = make_uv_sphere_exploded(**kwargs)
        self.vertex_count = pos.shape[0]

        # VAO/VBOs
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.vbo_pos = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_pos)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, pos.nbytes, pos, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        self.vbo_nor = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_nor)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, nor.nbytes, nor, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        self.vbo_bary = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_bary)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, bary.nbytes, bary, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        self.vbo_face = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_face)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, face.nbytes, face, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(3)
        # integer attribute
        GL.glVertexAttribIPointer(3, 1, GL.GL_UNSIGNED_INT, 0, None)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        # Shaders
        self.prog_beauty = link_program(BEAUTY_VS, BEAUTY_FS)
        self.prog_pick = link_program(PICK_VS, PICK_FS)
        return self
    def _ensure_pick_fbo(self, w, h):
        if self.pick_fbo is not None and self._fb_w == w and self._fb_h == h:
            return

        # delete old if exists
        if self.pick_fbo is not None:
            GL.glDeleteFramebuffers(1, [self.pick_fbo])
            GL.glDeleteTextures([self.pick_tex_obj, self.pick_tex_face, self.pick_tex_bary])
            GL.glDeleteRenderbuffers(1, [self.pick_depth])

        self._fb_w, self._fb_h = w, h

        self.pick_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.pick_fbo)

        # Object ID texture: R32UI
        self.pick_tex_obj = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.pick_tex_obj)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32UI, w, h, 0, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.pick_tex_obj, 0)

        # Face ID texture: R32UI
        self.pick_tex_face = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.pick_tex_face)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32UI, w, h, 0, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D, self.pick_tex_face, 0)

        # Barycentric: RG32F
        self.pick_tex_bary = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.pick_tex_bary)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RG32F, w, h, 0, GL.GL_RG, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D, self.pick_tex_bary, 0)

        # Depth buffer
        self.pick_depth = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.pick_depth)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, w, h)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.pick_depth)

        GL.glDrawBuffers(3, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2])

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Pick FBO incomplete: 0x{status:X}")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # ---- rendering ----
    def _render_pick_pass(self, w, h, eye, model, mvp):
        self._ensure_pick_fbo(w, h)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.pick_fbo)
        GL.glViewport(0, 0, w, h)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

        # clear: obj=0, face=0, bary=(0,0)
        GL.glClearBufferuiv(GL.GL_COLOR, 0, np.array([0], dtype=np.uint32))
        GL.glClearBufferuiv(GL.GL_COLOR, 1, np.array([0], dtype=np.uint32))
        GL.glClearBufferfv(GL.GL_COLOR, 2, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self.prog_pick)
        set_uniform_mat4(self.prog_pick, "uMVP", mvp)
        set_uniform_u1(self.prog_pick, "uObjectId", self.object_id)

        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _render_beauty_pass(self, w, h, eye, model, mvp, normal_mat):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, w, h)
        GL.glEnable(GL.GL_DEPTH_TEST)

        # REMOVE THIS (it clears for every object!)
        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self.prog_beauty)
        set_uniform_mat4(self.prog_beauty, "uMVP", mvp)
        set_uniform_mat4(self.prog_beauty, "uModel", model)
        set_uniform_mat3(self.prog_beauty, "uNormalMat", normal_mat)
        set_uniform_vec3(self.prog_beauty, "uEyeW", eye)
        set_uniform_vec3(self.prog_beauty, "uLightW", self.world.light_world)

        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def picked(self,x_tk,y_tk,x,y):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.pick_fbo)

        # Read object id
        obj = np.zeros((1,), dtype=np.uint32)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glReadPixels(x, y, 1, 1, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, obj)

        # Read face id
        face = np.zeros((1,), dtype=np.uint32)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        GL.glReadPixels(x, y, 1, 1, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, face)

        # Read bary (u,v)
        bary_uv = np.zeros((2,), dtype=np.float32)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
        GL.glReadPixels(x, y, 1, 1, GL.GL_RG, GL.GL_FLOAT, bary_uv)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        obj_id = int(obj[0])
        face_id = int(face[0])
        b0 = float(bary_uv[0])
        b1 = float(bary_uv[1])
        b2 = 1.0 - b0 - b1

        # Clamp tiny numerical overshoots
        b0, b1, b2 = (max(-1e-4, min(1.0, v)) for v in (b0, b1, b2))

        if obj_id == self.object_id:
            return self.object_id,face_id,(b0,b1,b2),x_tk,y_tk
        else:
            return None



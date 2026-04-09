from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from OpenGL import GL
from .Geometry import make_uv_sphere_exploded
from .Shader import BEAUTY_VS, BEAUTY_FS, PICK_VS, PICK_FS
from .gl_backend import set_uniform_mat4, set_uniform_mat3, set_uniform_vec3, set_uniform_u1, link_program,vao

if TYPE_CHECKING:
    from MVP import Three_D_Frame




class Object:
    def __init__(self,world:Three_D_Frame,obj_id):
                # Picking FBO (created on first draw / resize)
        self.world=world
        self.object_id=obj_id
        self.pick_render_update=lambda : None
        self.beauty_render_update=lambda : None
        self.center=np.array([0,0,0],dtype=np.float32)
        self.prog_pick = link_program(PICK_VS, PICK_FS)
        def temp_function():
            GL.glUseProgram(self.prog_pick)
            w = max(1, self.world.winfo_width())
            h = max(1, self.world.winfo_height())
            eye, model, view, proj, mvp, normal_mat = self.world._matrices(w, h)
            set_uniform_mat4(self.prog_pick, "uMVP", mvp)
            set_uniform_u1(self.prog_pick, "uObjectId", self.object_id)
        self.pick_render_update = temp_function

    def bind_sphere_default(self,**kwargs):
        pos, nor, bary, face, num_faces = make_uv_sphere_exploded(**kwargs)
        self.center=np.mean(pos, axis=0)
        self.vertex_count = pos.shape[0]
        vao_id, vbos = vao([pos, nor, bary, face], layout="3f 3f 3f 1u")
        self.vao = vao_id
        self.pos_vbo, self.nor_vbo, self.bary_vbo, self.face_vbo = vbos
        # Shaders
        self.prog_beauty = link_program(BEAUTY_VS, BEAUTY_FS)
        

        def temp_function():
            GL.glUseProgram(self.prog_beauty)
            w = max(1, self.world.winfo_width())
            h = max(1, self.world.winfo_height())
            eye, model, view, proj, mvp, normal_mat = self.world._matrices(w, h)
            set_uniform_mat4(self.prog_beauty, "uMVP", mvp)
            set_uniform_mat4(self.prog_beauty, "uModel", model)
            set_uniform_mat3(self.prog_beauty, "uNormalMat", normal_mat)
            set_uniform_vec3(self.prog_beauty, "uEyeW", eye)
            set_uniform_vec3(self.prog_beauty, "uLightW", self.world.light_world)

        self.beauty_render_update = temp_function
        return self
    
    def bind_custom(self,data:list[np.ndarray],VS:str,FS:str,beauty_render_update:function,layout:str="3f 3f 3f 1u",locs:list[int]|None=None):
        vao_id, vbos = vao(data, layout=layout, locs=locs)
        self.vao = vao_id
        self.prog_beauty = link_program(VS, FS)
        self.beauty_render_update = beauty_render_update
        self.beauty_render_update=beauty_render_update
        return self
    

    # ---- rendering ----
    def _render_pick_pass(self,mod=GL.GL_TRIANGLES):
        
        GL.glUseProgram(self.prog_pick)
        self.pick_render_update()  # allow caller to update any state before pick pass
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(mod, 0, self.vertex_count)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)


    def _render_beauty_pass(self,mod=GL.GL_TRIANGLES):
        # make sure to set program active befor calling the update function, so that uniform updates work
        GL.glUseProgram(self.prog_beauty)
        self.beauty_render_update()  # allow caller to update any state before beauty pass
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(mod, 0, self.vertex_count)
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



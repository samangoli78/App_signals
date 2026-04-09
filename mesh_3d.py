import tkinter as tk
from Engine.MVP import Three_D_Frame
from Engine.GL_Objects import Object
from App import CARTO_Tool
from Engine.Shader import mesh_vs,mesh_fs
from Engine.gl_backend import set_uniform_mat4, set_uniform_mat3, set_uniform_vec3, set_uniform_u1, link_program, vao,make_colormap_lut,upload_colormap_1d
import numpy as np
from OpenGL import GL


















if __name__ == "__main__":
    root = tk.Tk()
    root.title("Tkinter OpenGL Shader Sphere + ID/Barycentric Picking")

    frame = Three_D_Frame(root, width=960, height=720)
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
        obj3=Object(world=frame,obj_id=3)
        carto=CARTO_Tool.Carto()
        vertices,triangles,uni,bi,lat =carto.pars_mesh_file_with_electrode()
        verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        tris  = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
        scalars= np.asarray(lat, dtype=np.float32).reshape(-1)  # per-vertex scalar for colormap
        
        # ----- compute per-vertex normals from triangles -----
        normals = np.zeros_like(verts, dtype=np.float32)

        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]

        face_n = -np.cross(v1 - v0, v2 - v0)  # (F,3)

        # accumulate face normals to vertices
        np.add.at(normals, tris[:, 0], face_n)
        np.add.at(normals, tris[:, 1], face_n)
        np.add.at(normals, tris[:, 2], face_n)

        # normalize
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = -normals / np.maximum(nlen, 1e-20)

        
        frame.cmap_tex = upload_colormap_1d(make_colormap_lut(256))

        frame.scalar_min = float(np.min(scalars))
        frame.scalar_max = float(np.max(scalars))
        frame.n_levels   = 12   # discretization bands

        def temp_function():
            w = max(1, frame.winfo_width())
            h = max(1, frame.winfo_height())
            eye, model, view, proj, mvp, normal_mat = frame._matrices(w, h)

            prog = obj3.prog_beauty
            GL.glUseProgram(prog)  # REQUIRED before glUniform*

            # matrices
            set_uniform_mat4(prog, "m_proj", proj)
            set_uniform_mat4(prog, "m_view", view)
            set_uniform_mat4(prog, "m_model", model)
            set_uniform_mat3(prog, "m_normal", normal_mat)

            # scalar range + discretization
            GL.glUniform1f(GL.glGetUniformLocation(prog, "scalar_min"), frame.scalar_min)
            GL.glUniform1f(GL.glGetUniformLocation(prog, "scalar_max"), frame.scalar_max)
            GL.glUniform1i(GL.glGetUniformLocation(prog, "n_levels"), int(frame.n_levels))

            # colormap LUT (1D texture) on unit 0
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_1D, frame.cmap_tex)
            GL.glUniform1i(GL.glGetUniformLocation(prog, "colormap_tex"), 0)

            # lighting + appearance (match your mesh_fs)
            set_uniform_vec3(prog, "light_position", frame.light_world)
            set_uniform_vec3(prog, "light_color", np.array([1.0, 1.0, 1.0], np.float32))
            GL.glUniform1f(GL.glGetUniformLocation(prog, "ambient_intensity"), 0.25)
            GL.glUniform1f(GL.glGetUniformLocation(prog, "opacity"), 1.0)

        # IMPORTANT: triangles are NOT a vertex attribute.
        # Pass only per-vertex attributes: position, scalar, normal
        in_vert= []
        in_scalar = []
        in_normal = []
        for tri in triangles:
            for idx in tri:
                in_vert.append(verts[idx])
                in_scalar.append(scalars[idx])
                in_normal.append(normals[idx])
        in_vert = np.array(in_vert, dtype=np.float32)
        in_scalar = np.array(in_scalar, dtype=np.float32)
        in_normal = np.array(in_normal, dtype=np.float32)
        obj3.vertex_count = in_vert.shape[0]
        obj3.center = np.mean(in_vert, axis=0)
        obj3.bind_custom(
            [in_vert, in_scalar, in_normal],
            mesh_vs, mesh_fs,
            temp_function,
            "3f 1f 3f"
        )

        frame.objs = [obj1, obj2, obj3]
    root.after(100, command)  # Schedule the command to run after 100ms (after initgl has been called)
    root.mainloop()
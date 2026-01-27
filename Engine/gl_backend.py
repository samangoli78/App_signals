import OpenGL.GL as GL
# ----------------- GL utilities -----------------

def compile_shader(src, shader_type):
    sh = GL.glCreateShader(shader_type)
    GL.glShaderSource(sh, src)
    GL.glCompileShader(sh)
    ok = GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS)
    if not ok:
        log = GL.glGetShaderInfoLog(sh).decode("utf-8", errors="replace")
        raise RuntimeError(f"Shader compile failed:\n{log}")
    return sh

def link_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL.GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)
    ok = GL.glGetProgramiv(prog, GL.GL_LINK_STATUS)
    if not ok:
        log = GL.glGetProgramInfoLog(prog).decode("utf-8", errors="replace")
        raise RuntimeError(f"Program link failed:\n{log}")
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return prog

def set_uniform_mat4(prog, name, mat4):
    loc = GL.glGetUniformLocation(prog, name)
    GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, mat4)  # GL_TRUE because numpy is row-major

def set_uniform_mat3(prog, name, mat3):
    loc = GL.glGetUniformLocation(prog, name)
    GL.glUniformMatrix3fv(loc, 1, GL.GL_TRUE, mat3)

def set_uniform_vec3(prog, name, v):
    loc = GL.glGetUniformLocation(prog, name)
    GL.glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))

def set_uniform_u1(prog, name, u):
    loc = GL.glGetUniformLocation(prog, name)
    GL.glUniform1ui(loc, int(u))


from __future__ import annotations

import OpenGL.GL as GL
import numpy as np

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


def vao(datas: list[np.ndarray], layout: str="3f 3f 3f 1u", locs: list[int] | None = [0,1,2,3]):
    """
    datas: list of np arrays (one per attribute)
    layout: e.g. or "3f 3f 3f 1u" means 3 floats, 3 floats, 3 floats, 1 unsigned int
    locs: optional explicit attribute locations, e.g. [0,1,2,3]
          if None -> uses 0..N-1
    returns: (vao_id, vbos)  # keep vbos alive
    """
    tokens = layout.split()  # IMPORTANT: tokens, not characters

    if len(datas) != len(tokens):
        raise ValueError(f"datas({len(datas)}) != layout tokens({len(tokens)}): {tokens}")

    if locs is None:
        locs = list(range(len(tokens)))
    if len(locs) != len(tokens):
        raise ValueError(f"locs({len(locs)}) != layout tokens({len(tokens)}): {tokens}")

    vao_id = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_id)

    vbos = []
    for arr, tok, loc in zip(datas, tokens, locs):
        # accept "3f" or "u1" or "1u"
        tok = tok.strip().lower()

        comps = int(tok[:-1])
        kind = tok[-1]

        if kind == "f":
            arr = np.asarray(arr, dtype=np.float32)
        elif kind == "u":
            arr = np.asarray(arr, dtype=np.uint32)
        else:
            raise ValueError(f"Bad layout token '{tok}'. Use like '3f' or '1u'.")

        vbo = GL.glGenBuffers(1)
        vbos.append(vbo)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_STATIC_DRAW)

        GL.glEnableVertexAttribArray(loc)
        if kind == "f":
            GL.glVertexAttribPointer(loc, comps, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        else:
            GL.glVertexAttribIPointer(loc, comps, GL.GL_UNSIGNED_INT, 0, None)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)
    return vao_id, vbos


def fbo(w: int, h: int, layout: str):
    """
    Create FBO from layout string.

    layout example:
        "1u 1u 2f"
        "4f"
        "1f 1f"

    Tokens format:
        "<N><type>"

        N     = number of components (1–4)
        type  = 'f' (float) or 'u' (unsigned int)

    Mapping:
        1f → GL_R32F
        2f → GL_RG32F
        3f → GL_RGB32F
        4f → GL_RGBA32F
        1u → GL_R32UI

    Fragment shader must match attachment index:
        layout(location=0) out ...
        layout(location=1) out ...
        etc.

    Returns:
        (fbo_id, textures)
    """

    tokens = layout.split()

    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

    textures = []
    draw_bufs = []

    for i, tok in enumerate(tokens):
        comps = int(tok[:-1])
        kind  = tok[-1]

        if kind == 'f':
            if comps == 1:
                internal, fmt = GL.GL_R32F, GL.GL_RED
            elif comps == 2:
                internal, fmt = GL.GL_RG32F, GL.GL_RG
            elif comps == 3:
                internal, fmt = GL.GL_RGB32F, GL.GL_RGB
            elif comps == 4:
                internal, fmt = GL.GL_RGBA32F, GL.GL_RGBA
            typ = GL.GL_FLOAT

        elif kind == 'u':
            internal = GL.GL_R32UI
            fmt = GL.GL_RED_INTEGER
            typ = GL.GL_UNSIGNED_INT

        tex = GL.glGenTextures(1)
        textures.append(tex)

        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)

        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal,
                        w, h, 0, fmt, typ, None)
        
        """ GL_COLOR_ATTACHMENT0 = 0x8CE0
            GL_COLOR_ATTACHMENT1 = 0x8CE1
            GL_COLOR_ATTACHMENT2 = 0x8CE2
            GL_COLOR_ATTACHMENT3 = 0x8CE3 """

        att = GL.GL_COLOR_ATTACHMENT0 + i
        draw_bufs.append(att)

        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,
                                  att,
                                  GL.GL_TEXTURE_2D,
                                  tex, 0)

    GL.glDrawBuffers(len(draw_bufs), draw_bufs)

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    return fbo, textures




def make_colormap_lut(n: int) -> np.ndarray:
    """
    Returns (n,3) float32 in [0,1].
    """
    n = int(max(2, n))
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)

    # piecewise linear stops
    stops_t = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    stops_c = np.array([
        [0,0,1],  # blue
        [0,1,1],  # cyan
        [0,1,0],  # green
        [1,1,0],  # yellow
        [1,0,0],  # red
    ], dtype=np.float32)

    lut = np.empty((n,3), dtype=np.float32)
    for k in range(3):
        lut[:,k] = np.interp(t, stops_t, stops_c[:,k]).astype(np.float32)

    return lut

def upload_colormap_1d(lut_rgb: np.ndarray) -> int:
    """
    Upload LUT as GL_TEXTURE_1D (RGB32F). Returns texture id.
    """
    lut_rgb = np.asarray(lut_rgb, dtype=np.float32)
    assert lut_rgb.ndim == 2 and lut_rgb.shape[1] == 3

    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_1D, tex)

    GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)

    GL.glTexImage1D(
        GL.GL_TEXTURE_1D, 0,
        GL.GL_RGB32F,
        lut_rgb.shape[0], 0,
        GL.GL_RGB, GL.GL_FLOAT,
        lut_rgb
    )

    GL.glBindTexture(GL.GL_TEXTURE_1D, 0)
    return tex
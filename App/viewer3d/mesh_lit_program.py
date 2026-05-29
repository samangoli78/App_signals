"""Optional GL 3.2+ lit mesh program: geometry shader supplies triangle barycentrics
so the fragment shader can:

* **Three finite corners** — same as classic Gouraud: linear blend of LUT
  texture coordinates, then ``texture1D`` (matches fixed-function shading).
* **Two finite, one NaN** — pixels in the NaN corner's Voronoi cell (largest
  barycentric weight on that vertex) sample the no-data colour; elsewhere the
  two finite corners blend with weights renormalised over their edge.
* **One finite, two NaN** — pixels closest to the finite vertex get its colour;
  the rest get no-data.
"""

from __future__ import annotations

import traceback

_VS = r"""#version 150 compatibility

in vec3 vert;
in vec3 norm;
in vec2 st;

out vec3 vNormEye;
out float vU;
out float vOk;

void main(void) {
    vNormEye = gl_NormalMatrix * norm;
    vU = st.x;
    vOk = st.y;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(vert, 1.0);
}
"""

_GS = r"""#version 150 compatibility

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 vNormEye[];
in float vU[];
in float vOk[];

out vec3 gNormEye;
out vec3 gBary;
flat out vec3 gU123;
flat out vec3 gOk123;

void main(void) {
    vec3 U = vec3(vU[0], vU[1], vU[2]);
    vec3 OK = vec3(vOk[0], vOk[1], vOk[2]);
    for (int i = 0; i < 3; i++) {
        gNormEye = vNormEye[i];
        gU123 = U;
        gOk123 = OK;
        gBary = vec3(i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0);
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
"""

_FS = r"""#version 150 compatibility

in vec3 gNormEye;
in vec3 gBary;
flat in vec3 gU123;
flat in vec3 gOk123;

uniform sampler1D cmap;
uniform vec4 noDataColor;

float bary_argmax(void) {
    float m0 = gBary.x;
    float m1 = gBary.y;
    float m2 = gBary.z;
    if (m0 >= m1 && m0 >= m2)
        return 0.0;
    if (m1 >= m2)
        return 1.0;
    return 2.0;
}

float pick_texcoord(void) {
    float u0 = gU123.x, u1 = gU123.y, u2 = gU123.z;
    float o0 = gOk123.x, o1 = gOk123.y, o2 = gOk123.z;
    float l0 = gBary.x, l1 = gBary.y, l2 = gBary.z;
    float nfin = o0 + o1 + o2;

    /* Three finite values: classic Gouraud blend in LUT coordinate space. */
    if (nfin > 2.5)
        return l0 * u0 + l1 * u1 + l2 * u2;

    float am = bary_argmax();
    int idxMax = int(am);

    /* Zero finite */
    if (nfin < 0.5) {
        return -1.0;
    }
    /* One finite */
    if (nfin < 1.5) {
        int fi = (o0 > 0.5) ? 0 : ((o1 > 0.5) ? 1 : 2);
        if (idxMax == fi) {
            if (fi == 0)
                return u0;
            if (fi == 1)
                return u1;
            return u2;
        }
        return -1.0;
    }

    /* Two finite, one NaN: ni = NaN vertex index */
    int ni = (o0 < 0.5) ? 0 : ((o1 < 0.5) ? 1 : 2);
    if (idxMax == ni)
        return -1.0;
    int a = (ni == 0) ? 1 : 0;
    int b = (ni == 2) ? 1 : 2;
    float la = (a == 0) ? l0 : ((a == 1) ? l1 : l2);
    float lb = (b == 0) ? l0 : ((b == 1) ? l1 : l2);
    float ua = (a == 0) ? u0 : ((a == 1) ? u1 : u2);
    float ub = (b == 0) ? u0 : ((b == 1) ? u1 : u2);
    float s = la + lb;
    if (s < 1e-8)
        return -1.0;
    return (la * ua + lb * ub) / s;
}

void main(void) {
    float u = pick_texcoord();
    vec3 N = normalize(gNormEye);
    vec3 L = normalize(vec3(gl_LightSource[0].position));
    float nd = max(dot(N, L), 0.0);
    vec3 amb = vec3(gl_LightSource[0].ambient);
    vec3 diff = vec3(gl_LightSource[0].diffuse);
    vec3 base;
    if (u < 0.0 || u > 1.0)
        base = noDataColor.rgb;
    else
        base = texture(cmap, u).rgb;
    vec3 lit = base * (amb + diff * nd);
    gl_FragColor = vec4(lit, 1.0);
}
"""


def try_compile_mesh_lit_program() -> int | None:
    """Return linked program id, or ``None`` if GS pipeline is unavailable."""
    try:
        from OpenGL.GL import GL_FRAGMENT_SHADER, GL_GEOMETRY_SHADER, GL_VERTEX_SHADER
        from OpenGL.GL import glDeleteProgram, glGetAttribLocation, glUseProgram
        from OpenGL.GL.shaders import compileProgram, compileShader

        vs = compileShader(_VS, GL_VERTEX_SHADER)
        gs = compileShader(_GS, GL_GEOMETRY_SHADER)
        fs = compileShader(_FS, GL_FRAGMENT_SHADER)
        prog = compileProgram(vs, gs, fs)
        glUseProgram(0)
        # compileProgram may leave failed shaders; locations queried lazily
        loc_vert = glGetAttribLocation(prog, "vert")
        loc_norm = glGetAttribLocation(prog, "norm")
        loc_st = glGetAttribLocation(prog, "st")
        if loc_vert < 0 or loc_norm < 0 or loc_st < 0:
            glDeleteProgram(prog)
            return None
        return int(prog)
    except Exception:
        traceback.print_exc()
        return None


def attrib_locations(program: int) -> tuple[int, int, int]:
    from OpenGL.GL import glGetAttribLocation

    return (
        int(glGetAttribLocation(program, "vert")),
        int(glGetAttribLocation(program, "norm")),
        int(glGetAttribLocation(program, "st")),
    )


def uniform_no_data_location(program: int) -> int:
    from OpenGL.GL import glGetUniformLocation

    return int(glGetUniformLocation(program, "noDataColor"))


def uniform_cmap_location(program: int) -> int:
    from OpenGL.GL import glGetUniformLocation

    return int(glGetUniformLocation(program, "cmap"))

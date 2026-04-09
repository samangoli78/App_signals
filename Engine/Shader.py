BEAUTY_VS = r"""
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNor;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMat;

out vec3 vPosW;
out vec3 vNorW;

void main(){
    vec4 pw = uModel * vec4(aPos, 1.0);
    vPosW = pw.xyz;
    vNorW = normalize(uNormalMat * aNor);
    gl_Position = uMVP * (uModel * vec4(aPos, 1.0));

}
"""

BEAUTY_FS = r"""
#version 330 core
in vec3 vPosW;
in vec3 vNorW;

uniform vec3 uEyeW;
uniform vec3 uLightW;

out vec4 FragColor;

void main(){
    vec3 N = normalize(vNorW);
    vec3 L = normalize(uLightW - vPosW);
    vec3 V = normalize(uEyeW - vPosW);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 64.0);

    vec3 base = vec3(0.70, 0.85, 1.00);
    vec3 ambient = 0.12 * base;
    vec3 color = ambient + diff * base + 0.35 * spec * vec3(1.0);

    FragColor = vec4(color, 1.0);
}
"""

PICK_VS = r"""
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=2) in vec3 aBary;     // (1,0,0) (0,1,0) (0,0,1)
layout(location=3) in uint aFaceId;   // triangle id

uniform mat4 uMVP;

out vec3 vBary;
flat out uint vFaceId;

void main(){
    vBary = aBary;
    vFaceId = aFaceId;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

PICK_FS = r"""
#version 330 core
in vec3 vBary;
flat in uint vFaceId;

uniform uint uObjectId;

layout(location=0) out uint outObj;
layout(location=1) out uint outFace;
layout(location=2) out vec2 outBary; // store bary.x, bary.y ; bary.z = 1-x-y

void main(){
    outObj = uObjectId;
    outFace = vFaceId;
    outBary = vBary.xy;
}
"""



mesh_vs = r"""
#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in float in_scalar;   // <-- scalar
layout (location = 2) in vec3 in_normals;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

uniform float scalar_min;
uniform float scalar_max;

out float v_t;          // normalized scalar [0..1]
out vec3 frag_normal;
out vec3 frag_position;

float normalize_scalar(float s, float lo, float hi) {
    float den = max(1e-20, hi - lo);
    return clamp((s - lo) / den, 0.0, 1.0);
}

void main(){
    vec4 world_position = m_model * vec4(in_position, 1.0);
    frag_position = world_position.xyz;

    frag_normal = normalize(in_normals);

    v_t = normalize_scalar(in_scalar, scalar_min, scalar_max);

    gl_Position = m_proj * m_view * world_position;
}
"""

mesh_fs = r"""
#version 330 core

in float v_t;
in vec3 frag_normal;
in vec3 frag_position;

uniform vec3 light_position;
uniform vec3 light_color;
uniform float ambient_intensity;
uniform float opacity;

uniform int n_levels;          // discretization bands
uniform sampler1D colormap_tex;

out vec4 FragColor;

float discretize(float t, int levels) {
    t = clamp(t, 0.0, 1.0);
    if (levels <= 1) return t;
    float L = float(levels);
    // map to {0, 1/(L-1), ..., 1}
    return floor(t * L) / (L - 1.0);
}

vec3 scalar_to_rgb(float t, int levels) {
    float td = discretize(t, levels);
    return texture(colormap_tex, td).rgb;
}

void main(){
    vec3 normal = normalize(frag_normal);
    vec3 light_dir = normalize(light_position - frag_position);

    float diffuse = max(dot(normal, light_dir), 0.0);
    float intensity = mix(ambient_intensity, 1.0, diffuse);

    vec3 base_color = scalar_to_rgb(v_t, n_levels);

    vec3 final_color = mix(base_color * 0.5, base_color * light_color, intensity);
    FragColor = vec4(final_color, opacity);
}

"""
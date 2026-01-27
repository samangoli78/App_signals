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
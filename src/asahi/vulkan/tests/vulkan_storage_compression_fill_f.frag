/* Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(push_constant) uniform P { float scale; uint ones; } pc;
layout(location = 0) out vec4 o;
void main()
{
   uvec2 p = uvec2(gl_FragCoord.xy);
   uvec4 k = pc.ones != 0u ? uvec4(1u)
                           : (uvec4(p.x + 2u * p.y) + uvec4(0u, 5u, 10u, 15u)) % 64u;
   o = vec4(k) * pc.scale;
}

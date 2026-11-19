/* Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 */
#version 450
struct Rec { vec4 rect; uvec4 ci; vec4 pad[2]; };
layout(set = 0, binding = 0, std140) uniform R { Rec r[1024]; } u;
layout(location = 0) flat out uint ci;
void main()
{
   vec2 p = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1);
   Rec r = u.r[gl_InstanceIndex];
   gl_Position = vec4((r.rect.xy + p * r.rect.zw) * 2.0 - 1.0, 0.0, 1.0);
   ci = r.ci.x;
}

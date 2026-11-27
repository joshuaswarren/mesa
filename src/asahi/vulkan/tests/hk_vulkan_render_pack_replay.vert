#version 450
/* SPDX-License-Identifier: MIT */
layout(push_constant) uniform PC {
   ivec4 rect;
   vec4 color;
   vec2 scale;
   float z;
} pc;
void main()
{
   const int cx[6] = int[](0, 1, 0, 1, 1, 0);
   const int cy[6] = int[](0, 0, 1, 0, 1, 1);
   vec2 px = vec2(cx[gl_VertexIndex] != 0 ? pc.rect.z : pc.rect.x,
                  cy[gl_VertexIndex] != 0 ? pc.rect.w : pc.rect.y);
   gl_Position = vec4(px * pc.scale - 1.0, pc.z, 1.0);
}

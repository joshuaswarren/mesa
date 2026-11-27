#version 450
/* SPDX-License-Identifier: MIT */
layout(push_constant) uniform PC {
   ivec4 rect;
   vec4 color;
   vec2 scale;
   float z;
} pc;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main()
{
   o0 = pc.color;
   o1 = pc.color.bgra;
}

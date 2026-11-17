/* Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(set = 0, binding = 1, std140) uniform C { vec4 c[64]; } u;
layout(push_constant) uniform P { uint x; } pc;
layout(location = 0) flat in uint ci;
layout(location = 0) out vec4 o;
void main() { o = u.c[(ci ^ pc.x) & 63u]; }

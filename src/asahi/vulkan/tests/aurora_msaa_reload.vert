/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(push_constant) uniform PC { uint base; } pc;
layout(location = 0) flat out uint tid;
uint h(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
void main() {
   uint t = pc.base + uint(gl_VertexIndex) / 3u, c = uint(gl_VertexIndex) % 3u;
   uint a = h(t * 4u + 1u), b = h(t * 4u + 2u), d = h(t * 4u + 3u);
   vec2 A = vec2(float(a & 0xffffu), float(a >> 16)) / 32768.0 - 1.0;
   vec2 B = vec2(float(b & 0xffffu), float(b >> 16)) / 32768.0 - 1.0;
   vec2 dir = normalize(B - A + vec2(1e-3));
   vec2 n = vec2(-dir.y, dir.x) * (1.0 + float(d & 7u)) / 540.0;
   vec2 P = c == 0u ? A : (c == 1u ? B : B + n);
   gl_Position = vec4(P, 0.5, 1.0);
   tid = t;
}

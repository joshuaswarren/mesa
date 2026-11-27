/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(location = 0) flat in uint tid;
layout(location = 0) out vec4 o;
uint h(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
void main() {
   uint v = h(tid ^ 0x9e3779b9u);
   o = vec4(float((v & 3u) + 1u), float(((v >> 2) & 3u) + 1u), float(((v >> 4) & 3u) + 1u), 1.0) / 4096.0;
}

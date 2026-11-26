#version 450
/* Triangles from the vertex index: 0 is a full-screen background at depth
 * 0.9, the rest are small and nearer, placed by the seed. */
layout(push_constant) uniform P { uint seed; } p;
uint h(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
float r(uint x) { return float(h(x) & 0xffffffu) / 16777216.0; }
void main() {
   uint t = uint(gl_VertexIndex) / 3u, v = uint(gl_VertexIndex) % 3u;
   if (t == 0u) {
      vec2 q[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
      gl_Position = vec4(q[v], 0.9, 1);
      return;
   }
   uint s = t * 7919u + p.seed * 104729u;
   vec2 c = vec2(r(s), r(s + 1u)) * 2.2 - 1.1;
   float a = r(s + 2u) * 6.2831853 + float(v) * 2.0943951;
   gl_Position = vec4(c + 0.05 * vec2(cos(a), sin(a)), 0.1 + 0.75 * r(s + 3u), 1);
}

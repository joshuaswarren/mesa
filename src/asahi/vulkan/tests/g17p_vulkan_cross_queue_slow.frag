#version 450
/* Writes exactly pc.value after pc.iters dependent multiply-adds, so the
 * render takes long enough for an early reader to catch it unfinished. */
layout(push_constant) uniform PC { float value; uint iters; } pc;
layout(location = 0) out vec4 color;

void main()
{
   float acc = gl_FragCoord.x * 1e-3;
   for (uint i = 0u; i < pc.iters; ++i)
      acc = fract(acc * 1.0001 + 0.37);
   color = vec4(pc.value + (acc > 2.0 ? 1.0 : 0.0));
}

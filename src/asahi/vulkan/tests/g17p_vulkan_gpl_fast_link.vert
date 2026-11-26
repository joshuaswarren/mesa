#version 450
layout(location = 0) out vec4 colour;
void main()
{
   vec2 p = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1);
   colour = vec4(p, 0.5, 0.5);
   gl_Position = vec4(p * 4.0 - 1.0, 0.0, 1.0);
}

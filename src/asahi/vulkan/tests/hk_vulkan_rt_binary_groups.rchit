#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec4 payload;
hitAttributeEXT vec2 attribs;

void main()
{
   payload = vec4(attribs, 1.0, 1.0);
}

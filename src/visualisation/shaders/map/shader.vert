#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

out vec3 VertexPosition;
out vec3 VertexColor;

uniform mat4 model;
uniform mat4 projection;

void main()
{
    gl_Position = projection * model * vec4(position, 1.0);
    VertexPosition = position;
    VertexColor = color;
}
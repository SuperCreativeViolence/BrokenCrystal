#include "Object.h"
#include <stdio.h>

Object::Object()
{
	position = vec3();
	rotation = quat();
	scale = vec3(1);
}

void Object::Rotate(vec3 euler)
{
	Rotate(euler.x, euler.y ,euler.z);
}

void Object::Rotate(float x, float y, float z)
{
	key_pitch += radians(x);
	key_yaw += radians(y);
	key_roll += radians(z);

	UpdateView();
}

void Object::Translate(vec3 vector)
{
	position += vector;
	UpdateView();
}

void Object::Translate(float x, float y, float z)
{
	Translate(vec3(x, y, z));
}

void Object::Scale(vec3 vector)
{
	scale += vector;
	UpdateView();
}

void Object::Scale(float x, float y, float z)
{
	Scale(vec3(x, y, z));
}

void Object::UpdateView()
{
	mat4 translation_mat;
	translation_mat = translate(translation_mat, position);

	rotation = rotation * quat(vec3(key_pitch, key_yaw, key_roll));
	rotation = normalize(rotation);
	key_pitch = key_yaw = key_roll = 0;
	mat4 rotation_mat = mat4_cast(rotation);

	mat4 scale_mat;
	scale_mat = glm::scale(scale_mat, scale);

	view_matrix = translation_mat * rotation_mat * scale_mat;
}

#include "Object.h"
#include <stdio.h>

Object::Object()
{
	position = vec3();
	rotation = quat();
	scale = vec3(1);
}

void Object::SetRotation(const quat& rot)
{
	rotation = rot;
}

quat Object::GetRotation() const
{
	return rotation;
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

void Object::Translate(vec3 vector, bool isLocal)
{
	position += isLocal ? rotation * vector : vector;
	UpdateView();
}

void Object::Translate(float x, float y, float z, bool isLocal)
{
	Translate(vec3(x, y, z), isLocal);
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
	mat4 translation_mat = translate(-position);

	rotation = rotation * quat(vec3(key_pitch, key_yaw, key_roll));
	rotation = normalize(rotation);
	key_pitch = key_yaw = key_roll = 0;
	mat4 rotation_mat = transpose(toMat4(rotation));

	mat4 scale_mat;
	scale_mat = glm::scale(scale_mat, scale);

	view_matrix = scale_mat * rotation_mat * translation_mat;
}

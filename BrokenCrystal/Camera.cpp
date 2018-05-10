#include "Camera.h"

Camera::Camera()
{
	position = vec3(15, 15, 15);
	pitch = yaw = roll = 0;
}

void Camera::Rotate(vec3 euler)
{
	Rotate(euler.x, euler.y, euler.z);
}

void Camera::Rotate(float x, float y, float z)
{
	pitch += x;
	yaw += y;

	pitch = clamp(pitch, -90.0f, 90.0f);
	yaw = fmod(yaw, 360.0f);

	quat rotation = toQuat(eulerAngleYX(radians(yaw), radians(pitch)));
	SetRotation(rotation);
	isdirty_update = true;
}

void Camera::LookAt(vec3 pos)
{
	SetRotation(conjugate(toQuat(lookAt(position, pos, vec3(0, 1, 0)))));
	pitch = glm::pitch(rotation);
	yaw = glm::yaw(rotation);
	isdirty_update = true;
}

void Camera::LookAt(float x, float y, float z)
{
	LookAt(vec3(x, y, z));
}

void Camera::UpdateView()
{
	if (!isdirty_update)
		return;

	mat4 translation_mat = translate(-position);

	rotation = rotation * quat(vec3(key_pitch, key_yaw, key_roll));
	rotation = normalize(rotation);
	key_pitch = key_yaw = key_roll = 0;
	mat4 rotation_mat = transpose(toMat4(rotation));

	mat4 scale_mat;
	scale_mat = glm::scale(scale_mat, scale);

	view_matrix = scale_mat * rotation_mat * translation_mat;
	isdirty_update = false;
}

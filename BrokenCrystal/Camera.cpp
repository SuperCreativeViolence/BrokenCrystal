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

#include "Camera.h"

Camera::Camera() : 
	cameraPosition(10.0f, 5.0f, 0.0f),
	cameraTarget(0.0f, 0.0f, 0.0f),
	cameraDistance(15.0f),
	cameraPitch(20.0f),
	cameraYaw(0.0f),
	upVector(	0.0f, 1.0f, 0.0f),
	nearPlane(1.0f),
	farPlane(1000.0f)
{

}

void Camera::Rotate(float deltaX, float deltaY)
{
	cameraPitch -= deltaX;
	cameraYaw -= deltaY;
	cameraYaw = btFmod(cameraYaw, 360.0f);
	btClamp(cameraPitch, 10.0f, 90.0f);
	UpdateCamera();
}

void Camera::Zoom(float distance)
{
	cameraDistance -= distance;
	cameraDistance = fmax(cameraDistance, 15.0f);
	UpdateCamera();
}

void Camera::SetScreen(int w, int h)
{
	screenWidth = w;
	screenHeight = h;
}

void Camera::UpdateCamera()
{
	if (screenWidth == 0 && screenHeight == 0)
		return;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	float aspectRatio = screenWidth / (float) screenHeight;
	glFrustum(-aspectRatio * nearPlane, aspectRatio * nearPlane, -nearPlane, nearPlane, nearPlane, farPlane);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	float pitch = cameraPitch * 0.01745329f;
	float yaw = cameraYaw * 0.01745329f;

	btQuaternion rotation(upVector, yaw);
	btVector3 cameraPosition(0, 0, 0);

	cameraPosition[2] = -cameraDistance;
	btVector3 forward(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
	if (forward.length2() < SIMD_EPSILON)
	{
		forward.setValue(1.f, 0.f, 0.f);
	}

	btVector3 right = upVector.cross(forward);
	btQuaternion roll(right, -pitch);

	cameraPosition = btMatrix3x3(rotation) * btMatrix3x3(roll) * cameraPosition;
	cameraPosition[0] = cameraPosition.getX();
	cameraPosition[1] = cameraPosition.getY();
	cameraPosition[2] = cameraPosition.getZ();
	cameraPosition += cameraTarget;

	gluLookAt(cameraPosition[0], cameraPosition[1], cameraPosition[2], cameraTarget[0], cameraTarget[1], cameraTarget[2], upVector.getX(), upVector.getY(), upVector.getZ());
}
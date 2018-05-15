#include "Camera.h"
#include "erand48.h"

Camera::Camera() : 
	_cameraTempPosition(10.0, 5.0, 0.0),
	cameraTarget(0.0, 0.0, 0.0),
	cameraDistance(15.0),
	cameraPitch(20.0),
	cameraYaw(0.0),
	upVector(0.0, 1.0, 0.0),
	nearPlane(1.0),
	farPlane(1000.0)
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

btVector3 Camera::GetWorldPosition()
{
	return cameraPosition;
}

btVector3 Camera::GetPickingRay(int x, int y)
{
	float tanFov = 1.0f / nearPlane;
	float fov = btScalar(2.0) * btAtan(tanFov);

	btVector3 rayFrom = cameraPosition;
	btVector3 rayForward = (cameraTarget - cameraPosition);
	rayForward.normalize();
	rayForward *= farPlane;

	btVector3 ver = upVector;
	btVector3 hor = rayForward.cross(ver);
	hor.normalize();
	ver = hor.cross(rayForward);
	ver.normalize();
	hor *= 2.f * farPlane * tanFov;
	ver *= 2.f * farPlane * tanFov;

	btScalar aspect = screenWidth / (btScalar) screenHeight;

	hor *= aspect;
	btVector3 rayToCenter = rayFrom + rayForward;
	btVector3 dHor = hor * 1.f / float(screenWidth);
	btVector3 dVert = ver * 1.f / float(screenHeight);
	btVector3 rayTo = rayToCenter - 0.5f * hor + 0.5f * ver;
	rayTo += btScalar(x) * dHor;
	rayTo -= btScalar(y) * dVert;

	return rayTo;
}

btVector3 Camera::GetPickingRay(btVector3 pos)
{
	return GetPickingRay(pos[0], pos[1]);
}

void Camera::SetScreen(int w, int h)
{
	screenWidth = w;
	screenHeight = h;
}

Ray Camera::GetPathRay(int x, int y, bool jitter, unsigned short *Xi)
{
	float tanFov = 1.0f / nearPlane;
	float fov = btScalar(2.0) * btAtan(tanFov);

	btVector3 rayFrom = cameraPosition;
	btVector3 rayForward = (cameraTarget - cameraPosition);
	rayForward.normalize();
	rayForward *= farPlane;

	btVector3 ver = upVector;
	btVector3 hor = rayForward.cross(ver);
	hor.normalize();
	ver = hor.cross(rayForward);
	ver.normalize();
	hor *= 2.f * farPlane * fov;
	ver *= 2.f * farPlane * fov;

	btScalar aspect = screenWidth / (btScalar) screenHeight;

	hor *= aspect;
	btVector3 rayToCenter = rayFrom + rayForward;
	btVector3 dHor = hor * 1.f / float(screenWidth);
	btVector3 dVert = ver * 1.f / float(screenHeight);
	btVector3 rayTo = rayToCenter - 0.5f * hor + 0.5f * ver;
	rayTo += btScalar(x) * dHor;
	rayTo -= btScalar(y) * dVert;

	/*
	
	Todo : Jitter
	
	*/

	double xSpacing = 

	if (jitter)
	{

	}

	return Ray(cameraPosition, rayTo.normalize());
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
	btVector3 _cameraTempPosition(0, 0, 0);

	_cameraTempPosition[2] = -cameraDistance;
	btVector3 forward(_cameraTempPosition[0], _cameraTempPosition[1], _cameraTempPosition[2]);
	if (forward.length2() < SIMD_EPSILON)
	{
		forward.setValue(1.f, 0.f, 0.f);
	}

	btVector3 right = upVector.cross(forward);
	btQuaternion roll(right, -pitch);

	_cameraTempPosition = btMatrix3x3(rotation) * btMatrix3x3(roll) * _cameraTempPosition;
	_cameraTempPosition[0] = _cameraTempPosition.getX();
	_cameraTempPosition[1] = _cameraTempPosition.getY();
	_cameraTempPosition[2] = _cameraTempPosition.getZ();
	_cameraTempPosition += cameraTarget;
	cameraPosition = _cameraTempPosition;
	gluLookAt(_cameraTempPosition[0], _cameraTempPosition[1], _cameraTempPosition[2], cameraTarget[0], cameraTarget[1], cameraTarget[2], upVector.getX(), upVector.getY(), upVector.getZ());
}
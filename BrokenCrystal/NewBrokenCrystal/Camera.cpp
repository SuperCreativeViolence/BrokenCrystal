#include "Camera.h"

Camera::Camera() :
	position(10.0, 5.0, 0.0),
	target(0.0, 0.0, 0.0),
	distance(16.0),
	pitch(20.0),
	yaw(0.0),
	upVector(0.0, 1.0, 0.0),
	nearPlane(1.0),
	farPlane(1000.0)
{

}


Camera::~Camera()
{
}

void Camera::UpdateScreen(int w, int h)
{
	width = w;
	height = h;
}

void Camera::UpdateCamera()
{
	assert(scene > 0 && height > 0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	aspectRatio = width / (float) height;
	glFrustum(-aspectRatio * nearPlane, aspectRatio * nearPlane, -nearPlane, nearPlane, nearPlane, farPlane);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	float p = btRadians(pitch);
	float y = btRadians(yaw);

	btQuaternion rotation(upVector, y);

	btVector3 cameraPosition(0, 0, 0);
	cameraPosition[2] = -distance;

	btVector3 forward(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
	if (forward.length2() < SIMD_EPSILON)
	{
		forward.setValue(1.f, 0.f, 0.f);
	}

	btVector3 right = upVector.cross(forward);

	btQuaternion roll(right, -p);

	cameraPosition = btMatrix3x3(rotation) * btMatrix3x3(roll) * cameraPosition;
	cameraPosition += target;
	position = cameraPosition;
	gluLookAt(cameraPosition[0], cameraPosition[1], cameraPosition[2], target[0], target[1], target[2], upVector[0], upVector[1], upVector[2]);
}

void Camera::Rotate(float deltaX, float deltaY)
{
	pitch -= deltaX;
	yaw -= deltaY;
	yaw = btFmod(yaw, 360.0f);
	btClamp(pitch, 10.0f, 89.0f);
	UpdateCamera();
}

void Camera::Zoom(float delta)
{
	distance -= delta;
	distance = fmax(distance, 1.0f);
	UpdateCamera();
}

btVector3 Camera::GetScreenPosition(int x, int y)
{
	GLint viewport[4];
	GLdouble modelview[16];
	GLdouble projection[16];
	GLfloat winX, winY, winZ;
	GLdouble posX, posY, posZ;

	glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
	glGetDoublev(GL_PROJECTION_MATRIX, projection);
	glGetIntegerv(GL_VIEWPORT, viewport);

	winX = (float)x;
	winY = (float)viewport[3] - (float)y;
	glReadPixels(x, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);

	gluUnProject(winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);

	return btVector3(posX, posY, posZ);
}

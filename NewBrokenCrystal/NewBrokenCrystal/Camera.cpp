#include "Camera.h"

Camera::Camera() :
	position(10.0, 5.0, 0.0),
	target(0.0, 0.0, 0.0),
	distance(15.0),
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
	aspectRatio = width / (float)height;
	glFrustum(-aspectRatio * nearPlane, aspectRatio * nearPlane, -nearPlane, nearPlane, nearPlane, farPlane);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	float pitch = btRadians(pitch);
	float yaw = btRadians(yaw);

	btQuaternion rotation(upVector, yaw);

	btVector3 cameraPosition(0, 0, 0);
	cameraPosition[2] = -distance;

	btVector3 forward(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
	if (forward.length2() < SIMD_EPSILON)
	{
		forward.setValue(1.f, 0.f, 0.f);
	}

	btVector3 right = upVector.cross(forward);

	btQuaternion roll(right, -pitch);

	cameraPosition = btMatrix3x3(rotation) * btMatrix3x3(roll) * cameraPosition;

	position[0] = cameraPosition.getX();
	position[1] = cameraPosition.getY();
	position[2] = cameraPosition.getZ();
	position += target;

	gluLookAt(position[0], position[1], position[2], target[0], target[1], target[2], upVector[0], upVector[1], upVector[2]);
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

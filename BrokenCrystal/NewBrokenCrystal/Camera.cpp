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
	aspectRatio = w / (float)h * 1.5;
}

void Camera::UpdateCamera()
{
	assert(scene > 0 && height > 0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, width, height);
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
	direction = (target - position).normalize();
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

Ray Camera::GetRay(int x, int y, bool jitter, unsigned short *Xi)
{
	//GLint viewport[4];
	//GLdouble modelview[16];
	//GLdouble projection[16];
	//GLfloat winX, winY, winZ;
	//GLdouble posX, posY, posZ;

	//glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
	//glGetDoublev(GL_PROJECTION_MATRIX, projection);
	//glGetIntegerv(GL_VIEWPORT, viewport);

	//winX = (float)x;
	//winY = (float)viewport[3] - (float)y;  // Subtract The Current Mouse Y Coordinate 

	//glReadPixels(x, winY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);//Reads the depth buffer

	//gluUnProject(winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);
	//return Ray(position, (btVector3(posX, posY, posZ) - position).normalized());


	//double xJitter = 0;
	//double yJitter = 0;

	//double xSpacing = (2.0 * aspectRatio) / (double)width;
	//double ySpacing = (double)2. / height;

	//btVector3 xDirection = btVector3(0, 0, 1).cross(direction * -1).normalize();
	//btVector3 yDirection = xDirection.cross(direction).normalize();

	//if (jitter)
	//{
	//	xJitter = (erand48(Xi) * xSpacing) - xSpacing * 0.5;
	//	yJitter = (erand48(Xi) * ySpacing) - ySpacing * 0.5;
	//}

	//btVector3 pixel = position + direction * 2;
	//pixel = pixel - xDirection * aspectRatio + xDirection * ((x * 2 * aspectRatio)*(1.0 / width)) + btVector3(xJitter, 0, 0);
	//pixel = pixel + yDirection - yDirection * ((y * 2.0)*(1.0 / height) + yJitter);

	//return Ray(position, (pixel - position).normalize());

	float tanFov = 1.0f / nearPlane;
	float fov = btScalar(2.0) * btAtan(tanFov);
	double xJitter = 0;
	double yJitter = 0;

	double xSpacing = (2.0 * aspectRatio) / (double)width;
	double ySpacing = (double)2. / height;

	btVector3 rayFrom = position;
	btVector3 rayForward = (target - position);
	rayForward.normalize();
	rayForward *= farPlane;

	btVector3 ver = upVector;
	btVector3 hor = rayForward.cross(ver);
	hor.normalize();
	ver = hor.cross(rayForward);
	ver.normalize();
	hor *= 2.f * farPlane * fov;
	ver *= 2.f * farPlane * fov;

	if (jitter)
	{
		//xJitter = (erand48(Xi) * xSpacing) - xSpacing * 0.5;
		//yJitter = (erand48(Xi) * ySpacing) - ySpacing * 0.5;
		xJitter = erand48(Xi);
		yJitter = erand48(Xi);
	}

	hor *= aspectRatio;
	btVector3 rayToCenter = rayFrom + rayForward;
	btVector3 dHor = hor * 1.f / float(width);
	btVector3 dVert = ver * 1.f / float(height);
	btVector3 rayTo = rayToCenter - 0.5f * hor + 0.5f * ver;

	rayTo += btScalar(x + xJitter) * dHor;
	rayTo -= btScalar(y + yJitter) * dVert;

	return Ray(position, rayTo.normalize());
}

int Camera::GetWidht()
{
	return width;
}

int Camera::GetHeight()
{
	return height;
}

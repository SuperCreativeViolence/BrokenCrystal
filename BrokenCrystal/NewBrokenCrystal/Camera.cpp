#include "Camera.h"

Camera::Camera() :
	position(10.0, 5.0, 0.0),
	target(0.0, 5.0, 0.0),
	distance(16.0),
	fov(90),
	pitch(20.0),
	yaw(0.0),
	upVector(0.0, 1.0, 0.0),
	nearPlane(1.0),
	farPlane(1000.0),
	aperture(4),
	focalLength(16)
{

}

Camera::~Camera()
{
}

void Camera::UpdateScreen(int w, int h)
{
	width = w;
	height = h;
	aspectRatio = w / (float)h;
}

void Camera::UpdateCamera()
{
	assert(scene > 0 && height > 0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, width, height);
	//glFrustum(-aspectRatio * nearPlane, aspectRatio * nearPlane, -nearPlane, nearPlane, nearPlane, farPlane);
	gluPerspective(fov, aspectRatio, nearPlane, farPlane);
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
	btClamp(pitch, 1.0f, 89.0f);
	UpdateCamera();
}

void Camera::Zoom(float delta)
{
	distance -= delta;
	distance = fmax(distance, 1.0f);
	UpdateCamera();
}

void Camera::Fov(float delta)
{
	fov += delta;
	UpdateCamera();
}

Ray Camera::GetRay(int x, int y, bool jitter)
{
	const double r1 = 2.0 * erand48();
	const double r2 = 2.0 * erand48();

	double dx;
	if (r1 < 1.0)
		dx = sqrt(r1) - 1.0;
	else
		dx = 1.0 - sqrt(2.0 - r1);

	double dy;
	if (r2 < 1.0)
		dy = sqrt(r2) - 1.0;
	else
		dy = 1.0 - sqrt(2.0 - r2);

	btVector3 wDir = btVector3(-direction).normalize();
	btVector3 uDir = upVector.cross(wDir).normalize();
	btVector3 vDir = wDir.cross(-uDir);

	float top = tan(btRadians(fov * 0.5));
	float right = aspectRatio * top;
	float bottom = -top;
	float left = -right;

	float imPlaneUPos = left + (right - left)*(((float)x + dx + 0.5f) / (float)width);
	float imPlaneVPos = bottom + (top - bottom)*(((float)y + dy + 0.5f) / (float)height);

	return Ray(position, (imPlaneUPos*uDir + imPlaneVPos * vDir - wDir).normalize());
}

Ray Camera::GetRay(int x, int y, int sx, int sy, bool dof)
{
	const double r1 = 2.0 * erand48();
	const double r2 = 2.0 * erand48();

	double dx;
	if (r1 < 1.0)
		dx = sqrt(r1) - 1.0;
	else
		dx = 1.0 - sqrt(2.0 - r1);

	double dy;
	if (r2 < 1.0)
		dy = sqrt(r2) - 1.0;
	else
		dy = 1.0 - sqrt(2.0 - r2);

	btVector3 wDir = btVector3(-direction).normalize();
	btVector3 uDir = upVector.cross(wDir).normalize();
	btVector3 vDir = wDir.cross(-uDir);

	float top = tan(btRadians(fov * 0.5));
	float right = aspectRatio * top;
	float bottom = -top;
	float left = -right;

	float imPlaneUPos = left + (right - left)*(((float)x + sx + dx + 0.5f) / (float)width);
	float imPlaneVPos = bottom + (top - bottom)*(((float)y + sy + dy + 0.5f) / (float)height);

	Ray result = Ray(position, (imPlaneUPos*uDir + imPlaneVPos * vDir - wDir).normalize());

	if (dof)
	{
		double u1 = r1 - 1.0;
		double u2 = r2 - 1.0;

		double fac = (double)(2 * 3.14159265358979323846 * u2);

		btVector3 offset = aperture * btVector3(u1 * cos(fac), u1 * sin(fac), 0.0);
		btVector3 focalPlaneIntersection = result.origin + result.direction * (focalLength / direction.dot(result.direction));
		result.origin = result.origin + offset;
		result.direction = (focalPlaneIntersection - result.origin).normalize();
	}

	return result;
}

int Camera::GetWidht()
{
	return width;
}

int Camera::GetHeight()
{
	return height;
}

btVector3 Camera::GetPosition()
{
	return position;
}

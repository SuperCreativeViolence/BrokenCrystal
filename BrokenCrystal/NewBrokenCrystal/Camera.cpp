#include "Camera.h"

Camera::Camera() :
	position(10.0, 5.0, 0.0),
	target(0.0, 5.0, 0.0),
	distance(16.0),
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
	float tanFov = 1.0f / nearPlane;
	float fov = btScalar(1.3) * btAtan(tanFov);
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
	hor = hor.normalize();
	ver = hor.cross(rayForward);
	ver = ver.normalize();
	hor *= 2.f * farPlane * fov;
	ver *= 2.f * farPlane * fov;

	if (jitter)
	{
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

	float tanFov = 1.0f / nearPlane;
	float fov = btScalar(1.3) * btAtan(tanFov);

	btVector3 rayFrom = position;
	btVector3 rayForward = (target - position);
	rayForward.normalize();
	rayForward *= farPlane;

	btVector3 ver = upVector;
	btVector3 hor = rayForward.cross(ver);
	hor = hor.normalize();
	ver = hor.cross(rayForward);
	ver = ver.normalize();
	hor *= 2.f * farPlane * fov;
	ver *= 2.f * farPlane * fov;

	hor *= aspectRatio;
	btVector3 rayToCenter = rayFrom + rayForward;
	btVector3 dHor = hor * 1.f / float(width);
	btVector3 dVert = ver * 1.f / float(height);
	btVector3 rayTo = rayToCenter - 0.5f * hor + 0.5f * ver;

	rayTo += btScalar(x+dx) * dHor;
	rayTo -= btScalar(y+dy) * dVert;

	Ray result = Ray(position, rayTo.normalize());

	if (dof)
	{
		double u1 = (erand48() * 2.0) - 1.0;
		double u2 = (erand48() * 2.0) - 1.0;

		double fac = (double) (2 * 3.14159265358979323846 * u2);

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

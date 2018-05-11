#ifndef CAMERA_H
#define CAMERA_H

#include <gl/freeglut.h>
#include <BulletPhysics/btBulletDynamicsCommon.h>

class Camera
{
public:
	Camera();

	void UpdateCamera();
	void Rotate(float deltaX, float deltaY);
	void Zoom(float distance);

	void SetScreen(int w, int h);
private:
	btVector3 cameraPosition;
	btVector3 cameraTarget;
	float nearPlane;
	float farPlane;
	btVector3 upVector;
	float cameraDistance;
	float cameraPitch;
	float cameraYaw;
	int screenWidth;
	int screenHeight;

};

#endif
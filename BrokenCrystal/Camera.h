#ifndef CAMERA_H
#define CAMERA_H

#include <gl/freeglut.h>
#include <BulletPhysics/btBulletDynamicsCommon.h>
#include "InputManager.h"
#include "Ray.h"

class Camera
{
public:
	Camera();

	void UpdateCamera();
	void Rotate(float deltaX, float deltaY);
	void Zoom(float distance);
	btVector3 GetWorldPosition();
	btVector3 GetPickingRay(btVector3 pos);
	btVector3 GetPickingRay(int x, int y);
	void SetScreen(int w, int h);
	Ray GetPathRay(int x, int y, bool jitter, unsigned short *Xi);
private:
	btVector3 cameraPosition;
	btVector3 _cameraTempPosition;
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
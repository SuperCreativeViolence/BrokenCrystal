#ifndef CAMERA_H
#define CAMERA_H

#include <freeglut.h>
#include <btBulletDynamicsCommon.h>
#include <assert.h>
#include "Ray.h"
#include "erand48.h"

class Camera
{
public:
	Camera();
	~Camera();

	void UpdateScreen(int w, int h);
	void UpdateCamera();
	void Rotate(float deltaX, float deltaY);
	void Zoom(float delta);
	btVector3 GetScreenPosition(int x, int y);
	Ray GetRay(int x, int y, bool jitter, unsigned short *Xi);
	Ray GetRay(int x, int y, int sx, int sy);
	int GetWidht();
	int GetHeight();

private:
	btVector3 position;
	btVector3 target;
	btVector3 direction;
	float nearPlane;
	float farPlane;
	btVector3 upVector;
	float distance;
	float pitch;
	float yaw;

	float aspectRatio;

	int width;
	int height;
};

#endif
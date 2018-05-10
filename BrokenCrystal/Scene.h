#ifndef SCENE_H
#define SCENE_H
#include <BulletPhysics/btBulletDynamicsCommon.h>
#include "Camera.h"
#include "InputManager.h"
#include <gl/glut.h>

class Scene
{
public:
	Scene();
	typedef std::unique_ptr<Scene> p;
	static p Create() { return p(new(Scene)); }

	Camera::p camera;
	Object::p cube;
	void Update();
	void Render();

private:
	void DrawAxis(int size);
	void DrawGrid(float size, float step);
	void DrawDebugCube();
};

#endif
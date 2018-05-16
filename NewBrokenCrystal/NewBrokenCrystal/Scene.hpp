#ifndef SCENE_H
#define SCENE_H

#include <gl/freeglut.h>
#include <btBulletDynamicsCommon.h>

#include "Camera.h"

class Scene
{
public:
	Scene();
	~Scene();

	void Initialize();

	// opengl
	void Keyboard(unsigned char key, int x, int y);
	void KeyboardUp(unsigned char key, int x, int y);
	void Special(int key, int x, int y);
	void SpecialUp(int key, int x, int y);
	void Reshape(int w, int h);
	void Idle();
	void Mouse(int button, int state, int x, int y);
	void PassiveMotion(int x, int y);
	void Motion(int x, int y);
	void Display();

private:
	// 코어 Bullet Physics
	btBroadphaseInterface * broadphase;
	btCollisionConfiguration* collisionConfiguration;
	btCollisionDispatcher* dispatcher;
	btConstraintSolver* solver;
	btDynamicsWorld* world;

	//카메라
	Camera* camera;

};

#endif
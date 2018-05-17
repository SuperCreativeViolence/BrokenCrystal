#ifndef SCENE_H
#define SCENE_H

#include <gl/freeglut.h>
#include <btBulletDynamicsCommon.h>
#include <LinearMath/btQuickprof.h>
#include <LinearMath\btScalar.h>

#include "Camera.h"
#include "Object.h"

#include <vector>

typedef std::vector<Object*> Objects;

class Scene
{
public:
	Scene();
	~Scene();

	void Initialize();

	// object
	void AddObject(Object* object);
	void CreateBox(const btVector3 &position, const btVector3 &halfExtents, float mass, Material material);
	void CreateSphere(const btVector3 &position, double radius, float mass, Material material);

	// input
	bool IsKeyDown(unsigned char key);
	bool IsMouseDown(int mouse);

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

	// physics
	void UpdateScene(float dt);

	// opengl
	void RenderScene();
	void DrawShape(Object* object);
	void DrawBox(const btVector3& halfSize);
	void DrawSphere(float radius);

private:
	Objects objects;

	// input
	bool isMouseDrag;
	bool keyState[256];
	bool mouseState[3] = {1, 1, 1};
	int mousePos[2];
	int clickPos[2];
	int deltaDrag[2];

	// 코어 Bullet Physics
	btBroadphaseInterface * broadphase;
	btCollisionConfiguration* collisionConfiguration;
	btCollisionDispatcher* dispatcher;
	btConstraintSolver* solver;
	btDynamicsWorld* world;

	// physics
	btClock clock;

	// 카메라
	Camera* camera;

};

#endif
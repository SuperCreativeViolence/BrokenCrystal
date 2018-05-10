#ifndef SCENE_H
#define SCENE_H
#include <BulletPhysics/btBulletDynamicsCommon.h>
#include "Camera.h"
#include "InputManager.h"
#include <gl/freeglut.h>
#include "DebugDrawer.h"

#include <iostream>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

// Object 목록 관리를 위해
typedef std::vector<Object*> Objects;

// 충돌 이벤트와 관련
typedef std::pair<const btRigidBody*, const btRigidBody*> CollisionPair;
typedef std::set<CollisionPair> CollisionPairs;

class Scene
{
public:
	Scene();
	~Scene();
	typedef std::unique_ptr<Scene> p;
	static p Create() { return p(new(Scene)); }

	Camera::p camera;

	void Initialize();

	void Update();

	void Idle();
	void RenderScene();
	void UpdateScene(float deltaTime);
	void InitializePhysics();
	void ShutdownPhysics();
	void CreateObjects();

	Object* CreateGameObject(btCollisionShape* pShape, const float &mass,
		const btVector3 &color = btVector3(1.0f, 1.0f, 1.0f),
		const btVector3 &initialPosition = btVector3(0.0f, 0.0f, 0.0f),
		const btQuaternion &initialRotation = btQuaternion(0, 0, 1, 1));

	//충돌 이벤트 함수들
	void CheckForCollisionEvents();
	void CollisionEvent(btRigidBody* body0, btRigidBody * body1);
	void SeparationEvent(btRigidBody * body0, btRigidBody * body1);

private:
	void DrawAxis(int size);
	void DrawGrid(float size, float step);
	void DrawDebugCube();
	void DrawBox(const btVector3 &halfSize);
	void DrawShape(btScalar* transform, const btCollisionShape* pShape, const btVector3 &color);

	DebugDrawer* debugDrawer;
	Objects objects;
	btClock clock;

	// Bullet Physics 필수 컴포넌트
	btBroadphaseInterface* bt_Broadphase;
	btCollisionConfiguration* bt_CollisionConfiguration;
	btCollisionDispatcher* bt_Dispatcher;
	btConstraintSolver* bt_Solver;
	btDynamicsWorld* bt_World;

	// 충돌 관련 컴포넌트
	CollisionPairs collisionPairs_LastUpdate;
};

#endif
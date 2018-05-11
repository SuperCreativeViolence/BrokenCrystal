#ifndef SCENE_H
#define SCENE_H
#include <BulletPhysics/btBulletDynamicsCommon.h>
#include "Object.h"
#include "Camera.h"
#include "InputManager.h"
#include <gl/freeglut.h>
#include "DebugDrawer.h"

#include <iostream>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

// Draw 관련
#include <BulletPhysics/BulletCollision/CollisionShapes/btTriangleMeshShape.h>
#include <BulletPhysics/BulletCollision/CollisionShapes/btShapeHull.h>
#include <BulletPhysics/BulletCollision/CollisionShapes/btConvexPolyhedron.h>

// Object 목록 관리를 위해
typedef std::vector<Object*> Objects;

// 충돌 이벤트와 관련
typedef std::pair<const btRigidBody*, const btRigidBody*> CollisionPair;
typedef std::set<CollisionPair> CollisionPairs;


// 레이케스팅과 관련
struct RayResult
{
	btRigidBody* pBody;
	btVector3 hitPoint;
	btVector3 hitNormal;
};

class Scene
{
public:
	Scene();
	~Scene();
	typedef std::unique_ptr<Scene> p;
	static p Create() { return p(new(Scene)); }

	Camera* camera;

	void Initialize();

	void Update();

	void Idle();
	void Reshape(int w, int h);

	void RenderScene();
	void UpdateScene(float deltaTime);
	void InitializePhysics();
	void ShutdownPhysics();
	void CreateObjects();

	Object* CreateObject(btCollisionShape* pShape, const float &mass,
		const btVector3 &color = btVector3(1.0f, 1.0f, 1.0f),
		const btVector3 &initialPosition = btVector3(0.0f, 0.0f, 0.0f),
		const btQuaternion &initialRotation = btQuaternion(0, 0, 1, 1));

	Camera* CreateCamera();

	//충돌 이벤트 함수들
	void CheckForCollisionEvents();
	void CollisionEvent(btRigidBody* body0, btRigidBody * body1);
	void SeparationEvent(btRigidBody * body0, btRigidBody * body1);

	//레이케스팅
	bool RayCast(const btVector3 &start, const btVector3 &dir, RayResult &out, bool includeStatic = false);

private:
	void DrawAxis(int size);
	void DrawGrid(float size, float step);
	void DrawBox(const btVector3 &halfSize);
	void DrawSphere(btScalar radius, int lats, int longs);
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

	// 드로우 관련 (나중에 Drawer로 따로 빼자)
	struct ShapeCache
	{
		struct Edge
		{
			btVector3 n[2]; int v[2];
		};
		ShapeCache(btConvexShape* s) : m_shapehull(s)
		{
		}
		btShapeHull					m_shapehull;
		btAlignedObjectArray<Edge>	m_edges;
	};
	btAlignedObjectArray<ShapeCache*>	m_shapecaches;
	ShapeCache* cache(btConvexShape* shape);
};

#endif
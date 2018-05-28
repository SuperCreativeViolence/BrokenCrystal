#ifndef SCENE_H
#define SCENE_H
#define NOMINMAX
#include <gl/freeglut.h>
#include <btBulletDynamicsCommon.h>
#include <LinearMath/btQuickprof.h>
#include <LinearMath\btScalar.h>
#include <imgui_impl_glut.h>
#include <rbf.hpp>

#include "Camera.h"
#include "Object.h"
#include "lodepng.h"
#include "TracePath.h"
#include "voronoi.h"

#include <iostream>
#include <vector>

// Clamp double to min/max of 0/1
inline double clamp(double x) { return x < 0 ? 0 : x>1 ? 1 : x; }
inline int toInt(double x) { return int(clamp(x) * 255 + .5); }

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
	Mesh* CreateMesh(const btVector3 &position, const char* fileName, float mass, Material material);
	void DeleteObject(Object* object);

	// input
	bool IsKeyDown(unsigned char key);
	bool IsMouseDown(int mouse);

	// opengl
	void Keyboard(unsigned char key, int x, int y);
	void KeyboardUp(unsigned char key, int x, int y);
	void Special(int key, int x, int y);
	void SpecialUp(int key, int x, int y);
	void Reshape(int w, int h);
	void Mouse(int button, int state, int x, int y);
	void PassiveMotion(int x, int y);
	void Motion(int x, int y);
	void Display();
	void Idle();

	// physics
	void UpdateScene(float dt);

	// gui
	void RenderGUI();

	// opengl
	void RenderScene();
	void DrawShape(Object* object);
	void DrawBox(const btVector3& halfSize);
	void DrawSphere(float radius);
	void DrawTriangle(const btVector3 &p0, const btVector3 &p1, const btVector3 &p2);
	void DrawMesh(const Mesh* mesh);

	// path tracing
	void RenderPath(int samples);
	void RenderContinuousPath(int maxSamples = 1000000);
	btVector3 TraceRay(const Ray &ray, int depth);
	void DebugTraceRay(bool dof = false);
	btVector3 DebugPathTest(const Ray &ray, int depth, btVector3 hitPos);
	ObjectIntersection Intersect(const Ray &ray);
	void SaveImage(const char *filePath);

	// CUDA
	void CUMemInitialize();
	ObjectCU* CULoadObj(Object* object);
	void SaveImageCU(float3* pixels, const char *filePath);
	void DebugPathCU();
	void DrawMeshDebugCU();

private:
	Objects objects;

	// input
	bool isMouseDrag;
	bool keyState[256];
	bool mouseState[3] = {1, 1, 1};
	int mousePos[2];
	int clickPos[2];
	int deltaDrag[2];

	// cuda debug
	int cuda_mesh_flag;

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

	// path tracing
	int samples = 2;
	btVector3* pixelBuffer;

	// gui
	bool showDebugPanel = true;
	bool isTracing = false;
	float completion;
	int remaining;

};

#endif
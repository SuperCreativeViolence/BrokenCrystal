#include "Scene.hpp"


Scene::Scene() :
	world(nullptr),
	broadphase(nullptr),
	collisionConfiguration(nullptr),
	dispatcher(nullptr),
	solver(nullptr)
{
	camera = new Camera();
}

Scene::~Scene()
{
	delete camera;
	delete world;
	delete solver;
	delete broadphase;
	delete dispatcher;
	delete collisionConfiguration;
}

void Scene::Initialize()
{
	// opengl Light 초기화

	GLfloat ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f }; // dark grey
	GLfloat diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // white
	GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // white
	GLfloat position[] = { 5.0f, 10.0f, 1.0f, 0.0f };

	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
	glLightfv(GL_LIGHT0, GL_POSITION, position);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);

	glMaterialfv(GL_FRONT, GL_SPECULAR, specular);
	glMateriali(GL_FRONT, GL_SHININESS, 15);

	glShadeModel(GL_SMOOTH);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glClearColor(0.6, 0.65, 0.85, 0);

	// BulletPhysics 초기화

	collisionConfiguration = new btDefaultCollisionConfiguration();
	dispatcher = new btCollisionDispatcher(collisionConfiguration);
	broadphase = new btDbvtBroadphase();
	solver = new btSequentialImpulseConstraintSolver();
	world = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);
}

void Scene::Keyboard(unsigned char key, int x, int y)
{

}

void Scene::KeyboardUp(unsigned char key, int x, int y)
{
}

void Scene::Special(int key, int x, int y)
{

}

void Scene::SpecialUp(int key, int x, int y)
{

}

void Scene::Reshape(int w, int h)
{
	camera->UpdateScreen(w, h);
}

void Scene::Idle()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//float dt = m_clock.getTimeMilliseconds();
	//m_clock.reset();
	//UpdateScene(dt / 1000.0f);

	//UpdateCamera();

	//RenderScene();

	glutSwapBuffers();
}

void Scene::Mouse(int button, int state, int x, int y)
{
}

void Scene::PassiveMotion(int x, int y)
{
}

void Scene::Motion(int x, int y)
{
}

void Scene::Display()
{
}

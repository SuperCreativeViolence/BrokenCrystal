#include "Scene.h"


Scene::Scene() :
	world(nullptr),
	broadphase(nullptr),
	collisionConfiguration(nullptr),
	dispatcher(nullptr),
	solver(nullptr)
{
	std::fill_n(keyState, 256, 0);
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

	camera = new Camera();

	CreateBox(btVector3(0, 0, 0), btVector3(1, 500, 500), 0, Material());
	CreateSphere(btVector3(10, 10, 0), 3, 1, Material(DIFF, btVector3(0.3, 0.6, 0.1)));
	CreateSphere(btVector3(0, 4, 0), 1, 1, Material(DIFF, btVector3(0.3, 0.3, 0.1)));
	CreateBox(btVector3(0, 3, 3), btVector3(2,2,2), 1, Material(DIFF, btVector3(0.1, 0.2, 0.1)));
	CreateBox(btVector3(0, 2, 4), btVector3(2,2,2), 1, Material(DIFF, btVector3(0.6, 0.6, 0.1)));
	CreateBox(btVector3(2, 4, 0), btVector3(2,2,2), 1, Material(DIFF, btVector3(0.4, 0.9, 0.1)));
	CreateSphere(btVector3(0, 20, 0), 3, 0, Material(EMIT, btVector3(1, 1, 1), btVector3(3.3, 3.3, 3.3)));
}

void Scene::AddObject(Object* object)
{
	assert(world);
	objects.push_back(object);
	world->addRigidBody(object->GetRigidBody());
}

void Scene::CreateBox(const btVector3 &position, const btVector3 &halfExtents, float mass, Material material)
{
	AddObject(static_cast<Object*>(new Box(position, halfExtents, mass, material)));
}

void Scene::CreateSphere(const btVector3 &position, double radius, float mass, Material material)
{
	AddObject(static_cast<Object*>(new Sphere(position, radius, mass, material)));
}

bool Scene::IsKeyDown(unsigned char key)
{
	return keyState[key];
}

bool Scene::IsMouseDown(int mouse)
{
	return mouseState[mouse] == 0;
}

void Scene::Keyboard(unsigned char key, int x, int y)
{
	keyState[key] = true;
	mousePos[0] = x;
	mousePos[1] = y;
}

void Scene::KeyboardUp(unsigned char key, int x, int y)
{
	keyState[key] = false;
	mousePos[0] = x;
	mousePos[1] = y;
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

	float dt = clock.getTimeMilliseconds();
	clock.reset();
	UpdateScene(dt / 1000.0f);

	camera->UpdateCamera();
	RenderScene();

	glutSwapBuffers();
}

void Scene::Mouse(int button, int state, int x, int y)
{
	mouseState[button] = state;
	if (IsMouseDown(0))
	{
		glutSetCursor(GLUT_CURSOR_NONE);
		clickPos[0] = x;
		clickPos[1] = y;
	}
	else
	{
		glutSetCursor(GLUT_CURSOR_LEFT_ARROW);
	}
	mousePos[0] = x;
	mousePos[1] = y;
}

void Scene::PassiveMotion(int x, int y)
{
	mousePos[0] = x;
	mousePos[1] = y;
	deltaDrag[0] = 0;
	deltaDrag[1] = 0;
}

void Scene::Motion(int x, int y)
{
	mousePos[0] = x;
	mousePos[1] = y;
	deltaDrag[0] = 0;
	deltaDrag[1] = 0;
	isMouseDrag = true;
	if (IsMouseDown(0))
	{
		deltaDrag[0] = (clickPos[0] - x);
		deltaDrag[1] = (clickPos[1] - y);
		camera->Rotate(deltaDrag[1] * 0.1, deltaDrag[0] * 0.1);
		glutWarpPointer(clickPos[0], clickPos[1]);
		isMouseDrag = true;
	}
}

void Scene::Display()
{

}

void Scene::UpdateScene(float dt)
{
	assert(world);

	if (IsKeyDown('w'))
	{
		camera->Zoom(.5);
	}
	if (IsKeyDown('s'))
	{
		camera->Zoom(-.5);
	}

	world->stepSimulation(dt);
}

void Scene::RenderScene()
{
	for (Objects::iterator i = objects.begin(); i != objects.end(); i++)
	{
		DrawShape(*i);
	}
}

void Scene::DrawShape(Object* object)
{
	btScalar transform[16];
	btCollisionShape* pShape = object->GetShape();
	object->GetTransform(transform);

	glPushMatrix();
	glMultMatrixf(transform);
	glColor3fv(object->GetMaterial().GetColor());

	switch (pShape->getShapeType())
	{
		case BOX_SHAPE_PROXYTYPE:
		{
			const btBoxShape* box = static_cast<const btBoxShape*>(pShape);
			btVector3 halfSize = box->getHalfExtentsWithMargin();
			DrawBox(halfSize);
			break;
		}

		case SPHERE_SHAPE_PROXYTYPE:
		{
			const btSphereShape* sphere = static_cast<const btSphereShape*>(pShape);
			float radius = sphere->getMargin();
			DrawSphere(radius);
			break;
		}
	}
	glPopMatrix();
}

void Scene::DrawBox(const btVector3& halfSize)
{
	float halfWidth = halfSize.x();
	float halfHeight = halfSize.y();
	float halfDepth = halfSize.z();

	btVector3 vertices[8] =
	{ 
		btVector3(halfWidth, halfHeight, halfDepth),
		btVector3(-halfWidth, halfHeight, halfDepth),
		btVector3(halfWidth, -halfHeight, halfDepth),
		btVector3(-halfWidth, -halfHeight, halfDepth),
		btVector3(halfWidth, halfHeight, -halfDepth),
		btVector3(-halfWidth, halfHeight, -halfDepth),
		btVector3(halfWidth, -halfHeight, -halfDepth),
		btVector3(-halfWidth, -halfHeight, -halfDepth)
	};

	static int indices[36] =
	{ 
		0, 1, 2, 3, 2, 1, 4, 0, 6,
		6, 0, 2, 5, 1, 4, 4, 1, 0,
		7, 3, 1, 7, 1, 5, 5, 4, 7,
		7, 4, 6, 7, 2, 3, 7, 6, 2
	};

	glBegin(GL_TRIANGLES);

	for (int i = 0; i < 36; i += 3)
	{
		const btVector3 &vert1 = vertices[indices[i]];
		const btVector3 &vert2 = vertices[indices[i + 1]];
		const btVector3 &vert3 = vertices[indices[i + 2]];

		btVector3 normal = (vert3 - vert1).cross(vert2 - vert1);
		normal.normalize();

		glNormal3f(normal.getX(), normal.getY(), normal.getZ());

		glVertex3f(vert1.x(), vert1.y(), vert1.z());
		glVertex3f(vert2.x(), vert2.y(), vert2.z());
		glVertex3f(vert3.x(), vert3.y(), vert3.z());
	}

	glEnd();
}

void Scene::DrawSphere(float radius)
{
	int i, j;
	int lats = 50;
	int longs = 50;
	for (i = 0; i <= lats; i++)
	{
		btScalar lat0 = SIMD_PI * (-btScalar(0.5) + (btScalar) (i - 1) / lats);
		btScalar z0 = radius * sin(lat0);
		btScalar zr0 = radius * cos(lat0);

		btScalar lat1 = SIMD_PI * (-btScalar(0.5) + (btScalar) i / lats);
		btScalar z1 = radius * sin(lat1);
		btScalar zr1 = radius * cos(lat1);

		glBegin(GL_QUAD_STRIP);
		for (j = 0; j <= longs; j++)
		{
			btScalar lng = 2 * SIMD_PI * (btScalar) (j - 1) / longs;
			btScalar x = cos(lng);
			btScalar y = sin(lng);
			glNormal3f(x * zr1, y * zr1, z1);
			glVertex3f(x * zr1, y * zr1, z1);
			glNormal3f(x * zr0, y * zr0, z0);
			glVertex3f(x * zr0, y * zr0, z0);
		}
		glEnd();
	}
}

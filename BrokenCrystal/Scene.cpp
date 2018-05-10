#include "Scene.h"

Scene::Scene()
{
	camera = Camera::Create();
	InputManager::OnMouseDrag.permanent_bind([this](int deltaX, int deltaY) { this->camera->Rotate(deltaY * 0.1f, deltaX * 0.1f, 0); });
}

Scene::~Scene()
{
	ShutdownPhysics();
}

void Scene::Initialize()
{
	GLfloat ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };
	GLfloat diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
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

	InitializePhysics();

	debugDrawer = new DebugDrawer();
	debugDrawer->setDebugMode(0);
	bt_World->setDebugDrawer(debugDrawer);
}

void Scene::Update()
{
	if (InputManager::IsKeyDown('w'))
	{
		camera->Translate(0, 0, -1);
	}
	if (InputManager::IsKeyDown('s'))
	{
		camera->Translate(0, 0, 1);
	}
	if (InputManager::IsKeyDown('a'))
	{
		camera->Translate(-1, 0, 0);
	}
	if (InputManager::IsKeyDown('d'))
	{
		camera->Translate(1, 0, 0);
	}
	if (InputManager::IsKeyDown('q'))
	{
		camera->Translate(0, -1, 0);
	}
	if (InputManager::IsKeyDown('e'))
	{
		camera->Translate(0, 1, 0);
	}
}

void Scene::Idle()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	camera->UpdateView();


	float deltaTime = clock.getTimeMilliseconds();
	clock.reset();
	UpdateScene(deltaTime / 1000.0f);

	RenderScene();

	glutSwapBuffers();
}

void Scene::RenderScene()
{

	glMultMatrixf(value_ptr(camera->GetViewMatrix()));
	DrawGrid(50, 5);

	btScalar transform[16];

	for (Objects::iterator i = objects.begin(); i != objects.end(); ++i)
	{
		Object* obj = *i;
		obj->GetTransform(transform);
		DrawShape(transform, obj->GetShape(), obj->GetColor());
	}

	bt_World->debugDrawWorld();
}

void Scene::UpdateScene(float deltaTime)
{
	if (bt_World)
	{
		bt_World->stepSimulation(deltaTime);
		
		CheckForCollisionEvents();
	}
}

void Scene::InitializePhysics()
{
	bt_CollisionConfiguration = new btDefaultCollisionConfiguration();
	bt_Dispatcher = new btCollisionDispatcher(bt_CollisionConfiguration);
	bt_Broadphase = new btDbvtBroadphase();
	bt_Solver = new btSequentialImpulseConstraintSolver();
	bt_World = new btDiscreteDynamicsWorld(bt_Dispatcher, bt_Broadphase, bt_Solver, bt_CollisionConfiguration);

	CreateObjects();
}

void Scene::ShutdownPhysics()
{
	delete bt_World;
	delete bt_Solver;
	delete bt_Broadphase;
	delete bt_Dispatcher;
	delete bt_CollisionConfiguration;
}

void Scene::CreateObjects()
{
	CreateGameObject(new btBoxShape(btVector3(1, 50, 50)), 0, btVector3(0.2f, 0.6f, 0.6f), btVector3(0.0f, 0.0f, 0.0f));

	for (int i = 0; i < 10; i++)
	{
		CreateGameObject(new btBoxShape(btVector3(1, 1, 1)), 1.0, btVector3(1.0f, 0.2f, 0.2f), btVector3(0.0f, 10.0f * i, 0.0f));
	}

	CreateGameObject(new btBoxShape(btVector3(1, 1, 1)), 1.0, btVector3(1.0f, 0.2f, 0.2f), btVector3(0.0f, 10.0f, 0.0f));

	CreateGameObject(new btBoxShape(btVector3(1, 1, 1)), 1.0, btVector3(0.0f, 0.2f, 0.8f), btVector3(1.25f, 20.0f, 0.0f));

	//m_pTrigger = new btCollisionObject();
	//// create a box for the trigger's shape
	//m_pTrigger->setCollisionShape(new btBoxShape(btVector3(1, 0.25, 1)));
	//// set the trigger's position
	//btTransform triggerTrans;
	//triggerTrans.setIdentity();
	//triggerTrans.setOrigin(btVector3(0, 1.5, 0));
	//m_pTrigger->setWorldTransform(triggerTrans);
	//// flag the trigger to ignore contact responses
	//m_pTrigger->setCollisionFlags(btCollisionObject::CF_NO_CONTACT_RESPONSE);
	//bt_World->addCollisionObject(m_pTrigger);
}

Object* Scene::CreateGameObject(btCollisionShape* pShape, const float &mass, const btVector3 &color /*= btVector3(1.0f, 1.0f, 1.0f)*/, const btVector3 &initialPosition /*= btVector3(0.0f, 0.0f, 0.0f)*/, const btQuaternion &initialRotation /*= btQuaternion(0, 0, 1, 1)*/)
{
	Object* pObject = new Object(pShape, mass, color, initialPosition, initialRotation);

	objects.push_back(pObject);

	if (bt_World)
	{
		bt_World->addRigidBody(pObject->GetRigidBody());
	}
	return pObject;
}

void Scene::CheckForCollisionEvents()
{
	CollisionPairs pairsThisUpdate;

	for (int i = 0; i < bt_Dispatcher->getNumManifolds(); ++i)
	{
		btPersistentManifold* pManifold = bt_Dispatcher->getManifoldByIndexInternal(i);

		// 충돌하지 않은 도형은 체크안함
		if (pManifold->getNumContacts() > 0)
		{
			// 충돌한 두 RigidBody를 가지고옴
			const btRigidBody* body0 = static_cast<const btRigidBody*>(pManifold->getBody0());
			const btRigidBody* body1 = static_cast<const btRigidBody*>(pManifold->getBody1());

			bool const swapped = body0 > body1;
			const btRigidBody* pSortedBodyA = swapped ? body1 : body0;
			const btRigidBody* pSortedBodyB = swapped ? body0 : body1;

			CollisionPair thisPair = std::make_pair(pSortedBodyA, pSortedBodyB);

			pairsThisUpdate.insert(thisPair);

			// CollisionPair가 지난 충돌에서 보이지 않았다면
			// 새로운 충돌이 발생한 것
			if (collisionPairs_LastUpdate.find(thisPair) == collisionPairs_LastUpdate.end())
			{
				CollisionEvent((btRigidBody*)body0, (btRigidBody*)body1);
			}
		}
	}

	CollisionPairs removedPairs;

	// 지난 충돌과 현재 충돌을 비교
	std::set_difference(collisionPairs_LastUpdate.begin(), collisionPairs_LastUpdate.end(), pairsThisUpdate.begin(), pairsThisUpdate.end(), std::inserter(removedPairs, removedPairs.begin()));

	for (const auto & removedPair : removedPairs)
	{
		SeparationEvent((btRigidBody*)removedPair.first, (btRigidBody*)removedPair.second);
	}

	collisionPairs_LastUpdate = pairsThisUpdate;
}

void Scene::CollisionEvent(btRigidBody* body0, btRigidBody * body1)
{

}

void Scene::SeparationEvent(btRigidBody * body0, btRigidBody * body1)
{

}

void Scene::DrawAxis(int size)
{
	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);

	glColor3f(1, 0, 0);
	glVertex3f(0, 0, 0);
	glVertex3f(size, 0, 0);

	glColor3f(0, 1, 0);
	glVertex3f(0, 0, 0);
	glVertex3f(0, size, 0);

	glColor3f(0, 0, 1);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, size);
	glEnd();
	glEnable(GL_LIGHTING);
}

void Scene::DrawGrid(float size, float step)
{
	glDisable(GL_LIGHTING);

	glBegin(GL_LINES);

	glColor3f(0.3f, 0.3f, 0.3f);
	for (float i = step; i <= size; i += step)
	{
		glVertex3f(-size, 0, i);
		glVertex3f(size, 0, i);
		glVertex3f(-size, 0, -i);
		glVertex3f(size, 0, -i);

		glVertex3f(i, 0, -size);
		glVertex3f(i, 0, size);
		glVertex3f(-i, 0, -size);
		glVertex3f(-i, 0, size);
	}

	glColor3f(0.5f, 0, 0);
	glVertex3f(-size, 0, 0);
	glVertex3f(size, 0, 0);

	glColor3f(0, 0, 0.5f);
	glVertex3f(0, 0, -size);
	glVertex3f(0, 0, size);
	glEnd();
	glEnable(GL_LIGHTING);
}

void Scene::DrawBox(const btVector3 &halfSize)
{
	float halfWidth = halfSize.x();
	float halfHeight = halfSize.y();
	float halfDepth = halfSize.z();

	// Vertex 좌표 생성
	btVector3 vertices[8] =
	{
		btVector3(halfWidth, halfHeight, halfDepth),
		btVector3(-halfWidth,halfHeight, halfDepth),
		btVector3(halfWidth, -halfHeight,halfDepth),
		btVector3(-halfWidth, -halfHeight, halfDepth),
		btVector3(halfWidth, halfHeight, -halfDepth),
		btVector3(-halfWidth, halfHeight, -halfDepth),
		btVector3(halfWidth, -halfHeight, -halfDepth),
		btVector3(-halfWidth, -halfHeight, -halfDepth)
	};

	// 삼각형 꼭지점 생성
	static int indices[36] ={ 0, 1, 2, 3, 2, 1, 4, 0, 6, 6, 0, 2, 5, 1, 4, 4, 1, 0, 7, 3, 1, 7, 1, 5, 5, 4, 7, 7, 4, 6, 7, 2, 3, 7, 6, 2 };

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

void Scene::DrawShape(btScalar* transform, const btCollisionShape* pShape, const btVector3 &color)
{
	glColor3f(color.getX(), color.getY(), color.getZ());

	glPushMatrix();
	glMultMatrixf(transform);

	switch (pShape->getShapeType())
	{
		//Box 드로우
		case BOX_SHAPE_PROXYTYPE:
		{
			const btBoxShape* box = static_cast<const btBoxShape*>(pShape);
			btVector3 halfSize = box->getHalfExtentsWithMargin();
			DrawBox(halfSize);
			break;
		}
		default:
			break;
	}
	glPopMatrix();
}

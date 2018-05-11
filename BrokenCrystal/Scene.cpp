#include "Scene.h"

Scene::Scene()
{
	InputManager::OnMouseDrag.permanent_bind([this](int deltaX, int deltaY)
	{
		this->camera->Rotate(deltaY, deltaX);
	});
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
		camera->Zoom(1);
	}
	if (InputManager::IsKeyDown('s'))
	{
		camera->Zoom(-1);
	}
}

void Scene::Idle()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	float deltaTime = clock.getTimeMilliseconds();
	clock.reset();
	UpdateScene(deltaTime / 1000.0f);

	camera->UpdateCamera();

	RenderScene();

	glutSwapBuffers();
}

void Scene::Reshape(int w, int h)
{
	camera->SetScreen(w, h);
	glViewport(0, 0, w, h);
	camera->UpdateCamera();
}

void Scene::RenderScene()
{
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
	camera = CreateCamera();

	CreateObject(new btBoxShape(btVector3(0.01, 50, 50)), 0, btVector3(0.2f, 0.6f, 0.6f), btVector3(0.0f, -1.0f, 0.0f));

	for (int i = 0; i < 10; i++)
	{
		CreateObject(new btBoxShape(btVector3(1, 1, 1)), 1.0, btVector3(1.0f, 0.2f, 0.2f), btVector3(0.0f, 10.0f * i, 0.0f));
	}

	for (int i = 0; i < 10; i++)
	{
		CreateObject(new btSphereShape(1), 1.0, btVector3(1.0f, 0.2f, 0.2f), btVector3(3.0f, 10.0f * i, 3.0f));
	}

	CreateObject(new btBoxShape(btVector3(1, 1, 1)), 1.0, btVector3(1.0f, 0.2f, 0.2f), btVector3(0.0f, 10.0f, 0.0f));

	CreateObject(new btBoxShape(btVector3(1, 1, 1)), 1.0, btVector3(0.0f, 0.2f, 0.8f), btVector3(1.25f, 20.0f, 0.0f));
}

Object* Scene::CreateObject(btCollisionShape* pShape, const float &mass, const btVector3 &color /*= btVector3(1.0f, 1.0f, 1.0f)*/, const btVector3 &initialPosition /*= btVector3(0.0f, 0.0f, 0.0f)*/, const btQuaternion &initialRotation /*= btQuaternion(0, 0, 1, 1)*/)
{
	Object* pObject = new Object(pShape, mass, color, initialPosition, initialRotation);

	objects.push_back(pObject);

	if (bt_World)
	{
		bt_World->addRigidBody(pObject->GetRigidBody());
	}
	return pObject;
}

Camera* Scene::CreateCamera()
{
	Camera* pCamera = new Camera();

	//if (bt_World)
	//{
	//	bt_World->addRigidBody(pCamera->GetRigidBody());
	//}

	return pCamera;
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

void Scene::DrawSphere(btScalar radius, int lats, int longs)
{
	int i, j;
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

void Scene::DrawShape(btScalar* transform, const btCollisionShape* pShape, const btVector3 &color)
{
	glColor3f(color.getX(), color.getY(), color.getZ());

	glPushMatrix();
	glMultMatrixf(transform);

	switch (pShape->getShapeType())
	{
		// Box 드로우
		case BOX_SHAPE_PROXYTYPE:
		{
			const btBoxShape* box = static_cast<const btBoxShape*>(pShape);
			btVector3 halfSize = box->getHalfExtentsWithMargin();
			DrawBox(halfSize);
			break;
		}
		// Sphere 드로우
		case SPHERE_SHAPE_PROXYTYPE:
		{
			const btSphereShape* sphere = static_cast<const btSphereShape*>(pShape);
			float radius = sphere->getMargin();
			DrawSphere(radius, 15, 15);
			break;
		}
		// 나머지 btConvexHullShape 등 드로우 (메쉬)
		default:
		{
			if (pShape->isConvex())
			{
				const btConvexPolyhedron* poly = pShape->isPolyhedral() ? ((btPolyhedralConvexShape*) pShape)->getConvexPolyhedron() : 0;
				if (poly)
				{
					int i;
					glBegin(GL_TRIANGLES);
					for (i = 0; i < poly->m_faces.size(); i++)
					{
						btVector3 centroid(0, 0, 0);
						int numVerts = poly->m_faces[i].m_indices.size();
						if (numVerts > 2)
						{
							btVector3 v1 = poly->m_vertices[poly->m_faces[i].m_indices[0]];
							for (int v = 0; v < poly->m_faces[i].m_indices.size() - 2; v++)
							{

								btVector3 v2 = poly->m_vertices[poly->m_faces[i].m_indices[v + 1]];
								btVector3 v3 = poly->m_vertices[poly->m_faces[i].m_indices[v + 2]];
								btVector3 normal = (v3 - v1).cross(v2 - v1);
								normal.normalize();
								glNormal3f(normal.getX(), normal.getY(), normal.getZ());
								glVertex3f(v1.x(), v1.y(), v1.z());
								glVertex3f(v2.x(), v2.y(), v2.z());
								glVertex3f(v3.x(), v3.y(), v3.z());
							}
						}
					}
					glEnd();
				}
				else
				{
					ShapeCache*	sc = cache((btConvexShape*) pShape);
					btShapeHull* hull = &sc->m_shapehull;

					if (hull->numTriangles() > 0)
					{
						int index = 0;
						const unsigned int* idx = hull->getIndexPointer();
						const btVector3* vtx = hull->getVertexPointer();

						glBegin(GL_TRIANGLES);

						for (int i = 0; i < hull->numTriangles(); i++)
						{
							int i1 = index++;
							int i2 = index++;
							int i3 = index++;
							btAssert(i1 < hull->numIndices() &&
									 i2 < hull->numIndices() &&
									 i3 < hull->numIndices());

							int index1 = idx[i1];
							int index2 = idx[i2];
							int index3 = idx[i3];
							btAssert(index1 < hull->numVertices() &&
									 index2 < hull->numVertices() &&
									 index3 < hull->numVertices());

							btVector3 v1 = vtx[index1];
							btVector3 v2 = vtx[index2];
							btVector3 v3 = vtx[index3];
							btVector3 normal = (v3 - v1).cross(v2 - v1);
							normal.normalize();
							glNormal3f(normal.getX(), normal.getY(), normal.getZ());
							glVertex3f(v1.x(), v1.y(), v1.z());
							glVertex3f(v2.x(), v2.y(), v2.z());
							glVertex3f(v3.x(), v3.y(), v3.z());

						}
						glEnd();

					}
				}
			}
		}
	}
	glPopMatrix();
}

Scene::ShapeCache* Scene::cache(btConvexShape* shape)
{
	ShapeCache*		sc = (ShapeCache*) shape->getUserPointer();
	if (!sc)
	{
		sc = new(btAlignedAlloc(sizeof(ShapeCache), 16)) ShapeCache(shape);
		sc->m_shapehull.buildHull(shape->getMargin());
		m_shapecaches.push_back(sc);
		shape->setUserPointer(sc);
		/* Build edges	*/
		const int			ni = sc->m_shapehull.numIndices();
		const int			nv = sc->m_shapehull.numVertices();
		const unsigned int*	pi = sc->m_shapehull.getIndexPointer();
		const btVector3*	pv = sc->m_shapehull.getVertexPointer();
		btAlignedObjectArray<ShapeCache::Edge*>	edges;
		sc->m_edges.reserve(ni);
		edges.resize(nv*nv, 0);
		for (int i = 0; i < ni; i += 3)
		{
			const unsigned int* ti = pi + i;
			const btVector3		nrm = btCross(pv[ti[1]] - pv[ti[0]], pv[ti[2]] - pv[ti[0]]).normalized();
			for (int j = 2, k = 0; k < 3; j = k++)
			{
				const unsigned int	a = ti[j];
				const unsigned int	b = ti[k];
				ShapeCache::Edge*&	e = edges[btMin(a, b)*nv + btMax(a, b)];
				if (!e)
				{
					sc->m_edges.push_back(ShapeCache::Edge());
					e = &sc->m_edges[sc->m_edges.size() - 1];
					e->n[0] = nrm; e->n[1] = -nrm;
					e->v[0] = a; e->v[1] = b;
				}
				else
				{
					e->n[1] = nrm;
				}
			}
		}
	}
	return(sc);
}

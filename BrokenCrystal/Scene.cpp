#include "Scene.h"

Scene::Scene()
{
	InputManager::OnMouseDrag.permanent_bind([this](int deltaX, int deltaY)
	{
		this->camera->Rotate(deltaY, deltaX);
	});

	InputManager::OnMouseClick.permanent_bind([this](int mouse_event, int state, int x, int y)
	{
		switch (mouse_event)
		{
			case 0: // 왼쪽 마우스 버튼
			{
				if (state == 0)
				{

				}
				else
				{

				}
				break;
			}
			case 2: // 오른쪽 마우스 버튼
			{
				if (state == 0)
				{
					this->CreatePickingConstraint(x, y);
				}
				else
				{
					this->RemovePickingConstraint();
				}
				break;
			}

		}
	});

	world = 0;
	broadphase = 0;
	collisionConfiguration = 0;
	dispatcher = 0;
	solver = 0;
	pickedBody = 0;
	pickConstraint = 0;

	width = 0;
	height = 0;
	maxLevel = 0;
	antialiasing = false;
	pixels = nullptr;
}

Scene::Scene(int width, int height, int maxLevel, bool antialiasing) : Scene()
{
	this->width = width;
	this->height = height;
	this->maxLevel = maxLevel;
	this->antialiasing = antialiasing;
	pixels = new float[height * width * 3];
}

Scene::~Scene()
{
	delete camera;
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
	glDisable(GL_LIGHTING);

	InitializePhysics();
}

void Scene::Idle()
{
	//if (test)
	//{
	//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//	glDisable(GL_LIGHTING);
	//	glMatrixMode(GL_PROJECTION);
	//	glLoadIdentity();
	//	glViewport(0, 0, width, height);
	//	gluOrtho2D(0, width, 0, height);
	//	glMatrixMode(GL_MODELVIEW);
	//	glLoadIdentity();
	//	glRasterPos2i(0, 0);
	//	glDrawPixels(width, height, GL_RGB, GL_FLOAT, pixels);
	//	glMatrixMode(GL_MODELVIEW);

	//	glEnable(GL_LIGHTING);
	//	glutSwapBuffers();
	//}
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
	width = w;
	height = h;
	printf("%d %d\n", w, h);
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
		DrawShape(transform, obj->GetShape(), obj->GetMaterial().get_colour());
	}
	world->debugDrawWorld();
}

void Scene::UpdateScene(float deltaTime)
{
	if (world)
	{
		world->stepSimulation(deltaTime);
		
		CheckForCollisionEvents();
	}

	if (InputManager::IsKeyDown('w'))
	{
		camera->Zoom(1);
	}
	if (InputManager::IsKeyDown('s'))
	{
		camera->Zoom(-1);
	}
	if (InputManager::IsKeyDown('f'))
	{
		btVector3 mousePosition = InputManager::GetMousePos();
		ApplyCentralForce(mousePosition[0], mousePosition[1], 100);
	}
	if (InputManager::IsKeyDown('o'))
	{
		//RayTrace();
		PathTrace(samples);
		SaveImage("render.png");
		test = true;
	}

	if (pickedBody)
	{
		btGeneric6DofConstraint* pickCon = static_cast<btGeneric6DofConstraint*>(pickConstraint);
		if (!pickCon)
			return;
		btVector3 cameraPosition = camera->GetWorldPosition();
		btVector3 mousePosition = InputManager::GetMousePos();
		btVector3 dir = camera->GetPickingRay(mousePosition[0], mousePosition[1]) - cameraPosition;
		dir.normalize();

		dir *= oldPickingDistance;
		btVector3 newPivot = cameraPosition + dir;

		pickCon->getFrameOffsetA().setOrigin(newPivot);
	}
}

void Scene::InitializePhysics()
{
	collisionConfiguration = new btDefaultCollisionConfiguration();
	dispatcher = new btCollisionDispatcher(collisionConfiguration);
	broadphase = new btDbvtBroadphase();
	solver = new btSequentialImpulseConstraintSolver();
	world = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);
	CreateObjects();
}

void Scene::ShutdownPhysics()
{
	delete world;
	delete solver;
	delete broadphase;
	delete dispatcher;
	delete collisionConfiguration;
}

void Scene::CreateObjects()
{
	camera = new Camera();
	CreateSphere(btVector3(0, 5, 0), 3, Material(DIFF, btVector3(0.9, 0.9, 0.9)), 1.0);
	
	CreateSphere(btVector3(0, -1000, 0), 1000, Material(DIFF, btVector3(0.4, 0.2, 1.0)), 0);
	CreateSphere(btVector3(0, 10, 0), 5, Material(EMIT, btVector3(1.0, 1.0, 1.0), btVector3(2.2, 2.2, 2.2)), 0);
	//CreateSphere(btVector3(0, 5, 0), 5, Material(DIFF, btVector3(0.5, 0.5, 0)), 0);
	//CreateObject(new btBoxShape(btVector3(0.01, 50, 50)), 0, btVector3(0.2f, 0.6f, 0.6f), btVector3(0.0f, -1.0f, 0.0f));

	//for (int i = 0; i < 10; i++)
	//{
	//	CreateObject(new btBoxShape(btVector3(1, 1, 1)), 1.0, btVector3(1.0f, 0.2f, 0.2f), btVector3(0.0f, 10.0f * i, 0.0f));
	//}

	//for (int i = 0; i < 10; i++)
	//{
	//	CreateObject(new btSphereShape(1), 1.0, btVector3(1.0f, 0.2f, 0.2f), btVector3(3.0f, 10.0f * i, 3.0f));
	//}

	//CreateObject(new btBoxShape(btVector3(1, 1, 1)), 1.0, btVector3(1.0f, 0.2f, 0.2f), btVector3(0.0f, 10.0f, 0.0f));

	//CreateObject(new btBoxShape(btVector3(1, 1, 1)), 1.0, btVector3(0.0f, 0.2f, 0.8f), btVector3(1.25f, 20.0f, 0.0f));
}

//Object* Scene::CreateObject(btCollisionShape* pShape, const float &mass, const btVector3 &color /*= btVector3(1.0f, 1.0f, 1.0f)*/, const btVector3 &initialPosition /*= btVector3(0.0f, 0.0f, 0.0f)*/, const btQuaternion &initialRotation /*= btQuaternion(0, 0, 1, 1)*/)
//{
//	Object* pObject = new Object(pShape, mass, color, initialPosition, initialRotation);
//
//	objects.push_back(pObject);
//
//	if (world)
//	{
//		world->addRigidBody(pObject->GetRigidBody());
//	}
//	return pObject;
//}

void Scene::CheckForCollisionEvents()
{
	CollisionPairs pairsThisUpdate;

	for (int i = 0; i < dispatcher->getNumManifolds(); ++i)
	{
		btPersistentManifold* pManifold = dispatcher->getManifoldByIndexInternal(i);

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

bool Scene::RayCast(const btVector3 &start, const btVector3 &dir, RayResult &out, bool includeStatic /*= false*/)
{
	if (!world)
		return false;

	btVector3 rayTo = dir;
	btVector3 rayFrom = start;

	btCollisionWorld::ClosestRayResultCallback rayCallBack(rayFrom, rayTo);

	world->rayTest(rayFrom, rayTo, rayCallBack);

	if (rayCallBack.hasHit())
	{
		btRigidBody* pBody = (btRigidBody*) btRigidBody::upcast(rayCallBack.m_collisionObject);
		if (!pBody)
			return false;

		if (!includeStatic)
			if (pBody->isStaticObject() || pBody->isKinematicObject())
				return false;

		out.pBody = pBody;
		out.hitPoint = rayCallBack.m_hitPointWorld;
		out.hitNormal = rayCallBack.m_hitNormalWorld;
		return true;
	}
	return false;
}

void Scene::CreatePickingConstraint(int x, int y)
{
	if (!world)
		return;

	RayResult result;
	btVector3 cameraPosition = camera->GetWorldPosition();
	if (!RayCast(cameraPosition, camera->GetPickingRay(x, y), result))
		return;

	pickedBody = result.pBody;
	pickedBody->setActivationState(DISABLE_DEACTIVATION);
	btVector3 localPivot = pickedBody->getCenterOfMassTransform().inverse() * result.hitPoint;

	btTransform pivot = btTransform::getIdentity();
	pivot.setOrigin(localPivot);

	btGeneric6DofConstraint* dof6 = new btGeneric6DofConstraint(*pickedBody, pivot, true);
	dof6->setAngularLowerLimit(btVector3(0, 0, 0));
	dof6->setAngularUpperLimit(btVector3(0, 0, 0));

	world->addConstraint(dof6, true);
	pickConstraint = dof6;

	float cfm = 0.5f;
	dof6->setParam(BT_CONSTRAINT_STOP_CFM, cfm, 0);
	dof6->setParam(BT_CONSTRAINT_STOP_CFM, cfm, 1);
	dof6->setParam(BT_CONSTRAINT_STOP_CFM, cfm, 2);
	dof6->setParam(BT_CONSTRAINT_STOP_CFM, cfm, 3);
	dof6->setParam(BT_CONSTRAINT_STOP_CFM, cfm, 4);
	dof6->setParam(BT_CONSTRAINT_STOP_CFM, cfm, 5);

	float erp = 0.5f;
	dof6->setParam(BT_CONSTRAINT_STOP_ERP, erp, 0);
	dof6->setParam(BT_CONSTRAINT_STOP_ERP, erp, 1);
	dof6->setParam(BT_CONSTRAINT_STOP_ERP, erp, 2);
	dof6->setParam(BT_CONSTRAINT_STOP_ERP, erp, 3);
	dof6->setParam(BT_CONSTRAINT_STOP_ERP, erp, 4);
	dof6->setParam(BT_CONSTRAINT_STOP_ERP, erp, 5);

	oldPickingDistance = (result.hitPoint - cameraPosition).length();
}

void Scene::RemovePickingConstraint()
{
	if (!pickConstraint || !world)
		return;

	world->removeConstraint(pickConstraint);

	delete pickConstraint;

	pickedBody->forceActivationState(ACTIVE_TAG);
	pickedBody->setDeactivationTime(0.f);

	pickConstraint = 0;
	pickedBody = 0;
}

void Scene::ApplyCentralForce(int x, int y, float power)
{
	if (!world)
		return;

	RayResult result;
	btVector3 cameraPosition = camera->GetWorldPosition();
	if (RayCast(cameraPosition, camera->GetPickingRay(btVector3(x, y)), result))
	{
		btRigidBody* pickedBody = result.pBody;
		pickedBody->setActivationState(ACTIVE_TAG);
		pickedBody->applyCentralForce((result.hitPoint - cameraPosition).normalize() * power);
	}
}

void Scene::RayTrace()
{
	btVector3 cameraPosition = camera->GetWorldPosition();

	// 안티 계수 설정
	float inc = 1.0f, adj = 1.0f;
	if (antialiasing)
	{
		inc = 0.5f;
		adj = inc * inc;
	}
	// 스크린 픽셀 순회
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			btVector3 color(0, 0, 0);
			btVector3 screenPosition = camera->GetPickingRay(i, height - j);
			color += Trace(cameraPosition, screenPosition, 0);
			color *= adj;
			pixels[AT_R(i, j)] = color[0];
			pixels[AT_G(i, j)] = color[1];
			pixels[AT_B(i, j)] = color[2];
		}
	}
}

btVector3 Scene::Trace(const btVector3 &from, const btVector3 &to, int level)
{
	if (level >= maxLevel)
		return btVector3(0, 0, 0);

	btCollisionWorld::ClosestRayResultCallback rayCallBack(from, to);
	world->rayTest(from, to, rayCallBack);

	if (rayCallBack.hasHit())
	{
		return btVector3(0.4, 0.3, 0.2);
	}
	else
	{
		return btVector3(0, 0, 0);
	}
}

void Scene::PathTrace(int samples)
{
	double samples_recp = 1.0 / samples;
	pixel_buffer = new btVector3[width * height];
#pragma omp parallel for schedule(dynamic, 1)
	for (int y = 0; y < height; y++)
	{
		unsigned short Xi[3] = { 0, 0, y*y*y };
		fprintf(stderr, "\rRendering (%i samples): %.2f%% ", samples, (double) y / height * 100);

		for (int x = 0; x < width; x++)
		{
			btVector3 color = btVector3(0, 0, 0);

			for (int s = 0; s < samples; s++)
			{
				Ray ray = camera->GetPathRay(x, y, s > 0, Xi);
				color = color + TracePath(ray, 0, Xi);
				//printf("%f, %f, %f\n", color[0], color[1], color[2]);
				//Sleep(1000);
			}
			pixel_buffer[(y) *width + x] = color * samples_recp;
			//printf("%.2f, %.2f, %.2f\n", pixel_buffer[(y) *width + x][0], pixel_buffer[(y) *width + x][1], pixel_buffer[(y) *width + x][2]);
		}
	}
}

btVector3 Scene::TracePath(const Ray &ray, int depth, unsigned short *Xi)
{
	ObjectIntersection isct = intersect(ray);

	if (!isct.hit)
	{
		//printf("Return : 0.2, 0.2, 0.2 안맞음\n");
		return btVector3(0.2, 0.2, 0.2);
	}

	if (isct.m.get_type() == EMIT)
	{
		//printf("Return : %f, %f, %f 이미션에 맞음\n", isct.m.get_emission()[0], isct.m.get_emission()[1], isct.m.get_emission()[2]);
		return isct.m.get_emission();
	}
	btVector3 color = isct.m.get_colour();
	double p = color.x() > color.y() && color.x() > color.z() ? color.x() : color.y() > color.z() ? color.y() : color.z();
	double rnd = erand48(Xi);
	if (++depth > 5)
	{
		if (rnd < p * 0.9)
		{
			color = color * (0.9 / p);
		}
		else
		{
			//printf("Return : %f, %f, %f 뎁스 5 이상에서 이미션 리턴함\n", isct.m.get_emission()[0], isct.m.get_emission()[1], isct.m.get_emission()[2]);
			return isct.m.get_emission();
		}
	}

	btVector3 x = ray.origin + ray.direction * isct.u;
	Ray reflected = isct.m.get_reflected_ray(ray, x, isct.n, Xi);
	//printf("Return : %f, %f, %f 정상 리턴\n", color[0], color[1], color[2]);
	return color * TracePath(reflected, depth, Xi);
}

ObjectIntersection Scene::intersect(const Ray &ray)
{
	ObjectIntersection isct = ObjectIntersection();
	ObjectIntersection temp;
	long size = objects.size();

	for (int i = 0; i < size; i++)
	{
		temp = objects.at((unsigned) i)->get_intersection(ray);

		if (temp.hit)
		{
			if (isct.u == 0 || temp.u < isct.u) isct = temp;
		}
	}
	return isct;
}

inline double clamp(double x)
{
	return x < 0 ? 0 : x>1 ? 1 : x;
}

inline int toInt(double x)
{
	return int(clamp(x) * 255 + .5);
}

void Scene::SaveImage(const char *file_path)
{
	std::vector<unsigned char> buffer;

	int pixel_count = width * height;

	for (int i = 0; i < pixel_count; i++)
	{
		buffer.push_back(toInt(pixel_buffer[i].x()));
		buffer.push_back(toInt(pixel_buffer[i].y()));
		buffer.push_back(toInt(pixel_buffer[i].z()));
		buffer.push_back(255);
	}

	unsigned error = lodepng::encode(file_path, buffer, width, height);
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	buffer.clear();
}

void Scene::CreateSphere(const btVector3& position, const float& radius, const Material& material, const float &mass)
{
	Object * object = dynamic_cast<Object*>(new Sphere(position, radius, mass, material));
	objects.push_back(object);
	if (world)
	{
		world->addRigidBody(object->GetRigidBody());
	}
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

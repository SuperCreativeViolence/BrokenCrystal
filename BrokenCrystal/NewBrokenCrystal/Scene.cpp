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

	CreateBox(btVector3(0, 0, 0), btVector3(30, 1, 30), 0, Material(DIFF, btVector3(0.8, 0.8, 0.8)));
	CreateBox(btVector3(0, 30, 0), btVector3(30, 1, 30), 0, Material(EMIT, btVector3(1.0, 1.0, 1.0), btVector3(2.2, 2.2, 2.2)));
	CreateBox(btVector3(30, 15, 0), btVector3(1, 15, 30), 0, Material(DIFF, btVector3(0.0, 0.0, 0.85)));
	CreateBox(btVector3(-30, 15, 0), btVector3(1, 15, 30), 0, Material(DIFF, btVector3(0.85, 0.0, 0.0)));
	CreateBox(btVector3(0, 15, 30), btVector3(30, 15, 1), 0, Material(DIFF, btVector3(0.8, 0.8, 0.8)));
	CreateBox(btVector3(0, 15, -30), btVector3(30, 15, 1), 0, Material(DIFF, btVector3(0.8, 0.8, 0.8)));

	CreateSphere(btVector3(10, 10, 0), 2, 1, Material(SPEC, btVector3(1.0, 1.0, 1.0)));
	CreateSphere(btVector3(0, 4, 0), 2, 1, Material(DIFF, btVector3(0.3, 0.3, 0.1)));
	CreateSphere(btVector3(0, 10, 10), 2, 1, Material(SPEC, btVector3(1.0, 1.0, 1.0)));
	CreateSphere(btVector3(-3, 4, 4), 4, 1, Material(DIFF, btVector3(0.3, 0.1, 0.3)));

	CreateBox(btVector3(0, 3, 3), btVector3(2, 2, 2), 1, Material(DIFF, btVector3(0.1, 0.2, 0.1)));
	CreateBox(btVector3(0, 2, -4), btVector3(2, 2, 2), 1, Material(SPEC, btVector3(1.0, 1.0, 1.0)));
	CreateBox(btVector3(2, 4, 0), btVector3(2, 2, 2), 0, Material(DIFF, btVector3(0.4, 0.3, 0.1)));

	CreateMesh(btVector3(0, 5, 0), "dragon2.obj", 1, Material(SPEC, btVector3(1.0, 1.0, 1.0)));
}

void Scene::AddObject(Object* object)
{
	assert(world);
	objects.push_back(object);
	world->addRigidBody(object->GetRigidBody());
}

void Scene::CreateBox(const btVector3 &position, const btVector3 &halfExtents, float mass, Material material)
{
	float halfWidth = halfExtents[0];
	float halfHeight = halfExtents[1];
	float halfDepth = halfExtents[2];

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

	std::vector<Triangle*> triangles;

	for (int i = 0; i < 36; i += 3)
	{
		triangles.push_back(new Triangle(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]], material));
	}


	AddObject(static_cast<Object*>(new Mesh(position, triangles, mass, material)));
}

void Scene::CreateSphere(const btVector3 &position, double radius, float mass, Material material)
{
	AddObject(static_cast<Object*>(new Sphere(position, radius, mass, material)));
}

void Scene::CreateMesh(const btVector3 &position, const char* fileName, float mass, Material material)
{
	AddObject(static_cast<Object*>(new Mesh(position, fileName, mass, material)));
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

	camera->UpdateCamera();
	RenderScene();

	UpdateScene(dt / 1000.0f);

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
		camera->Zoom(.1);
	}
	if (IsKeyDown('s'))
	{
		camera->Zoom(-.1);
	}
	if (IsKeyDown('r'))
	{
		RenderPath(samples);
		SaveImage("Render.png");
	}
	if (IsKeyDown('d'))
	{
		DebugTraceRay();
	}
	if (IsKeyDown('f'))
	{
		Ray ray = camera->GetRay(mousePos[0], mousePos[1], true, nullptr);
		system("cls");
		btVector3 color = DebugPathTest(ray, 0, ray.origin);
		printf("\nresult = %.1f %.1f %.1f\n", color[0], color[1], color[2]);
	}

	world->stepSimulation(dt);
}

void Scene::RenderScene()
{
	for (auto & object : objects)
	{
		DrawShape(object);
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

		case CONVEX_TRIANGLEMESH_SHAPE_PROXYTYPE:
		{
			const Mesh* mesh = static_cast<const Mesh*>(object);
			DrawMesh(mesh->GetTriangles());
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

void Scene::DrawTriangle(const btVector3 &p0, const btVector3 &p1, const btVector3 &p2)
{
	btVector3 normal = (p2 - p0).cross(p1 - p0);
	normal = normal.normalize();
	glBegin(GL_TRIANGLE_STRIP);
	glNormal3fv(normal);
	glVertex3fv(p0);
	glVertex3fv(p1);
	glVertex3fv(p2);
	glEnd();
}

void Scene::DrawMesh(std::vector<Triangle*> triangles)
{
	for (auto & triangle : triangles)
	{
		DrawTriangle(triangle->pos[0], triangle->pos[1], triangle->pos[2]);
	}
}

void Scene::RenderPath(int samples)
{
	int width = camera->GetWidht();
	int height = camera->GetHeight();
	double samplesP = 1.0 / samples;
	pixelBuffer = new btVector3[width * height];

#pragma omp parallel for schedule(dynamic, 1)
	for (int y = 0; y < height; y++)
	{
		unsigned short Xi[3] = { 0, 0, y*y*y };
		fprintf(stderr, "\rRendering (%i samples): %.2f%% ", samples, (double)y / height * 100);

		for (int x = 0; x < width; x++)
		{
			btVector3 color = btVector3(0, 0, 0);

			for (int s = 0; s < samples; s++)
			{
				Ray ray = camera->GetRay(x, y, s > 0, Xi);
				color = color + TraceRay(ray, 0, Xi);
				//printf("%f %f %f\n", color[0], color[1], color[2]);
				//Sleep(1000);
			}
			//printf("final : %f %f %f\n", color[0] * samplesP, color[1] * samplesP, color[2] * samplesP);
			pixelBuffer[(y)* width + x] = color * samplesP;
		}
	}
}


btVector3 Scene::TraceRay(const Ray &ray, int depth, unsigned short *Xi)
{
	ObjectIntersection intersection = Intersect(ray);
	if (!intersection.hit) return btVector3(0.0, 0.0, 0.0);
	if (intersection.material.GetType() == EMIT)
		return intersection.material.GetEmission();

	btVector3 color = intersection.material.GetColor();
	double maxReflection = color.x()>color.y() && color.x()>color.z() ? color.x() : color.y()>color.z() ? color.y() : color.z();
	double random = erand48(Xi);

	if (++depth > 5)
	{
		if (random < maxReflection * 0.9)
		{
			color = color * (0.9 / maxReflection);
		}
		else
		{
			return intersection.material.GetEmission();
		}
	}

	btVector3 pos = ray.origin + ray.direction * intersection.u;
	Ray reflected = intersection.material.GetReflectedRay(ray, pos, intersection.normal, Xi);
	return color * TraceRay(reflected, depth, Xi);
}

void Scene::DebugTraceRay()
{
	unsigned short Xi[3] = { 0, 0, mousePos[1] * mousePos[1] * mousePos[1] };
	Ray ray = camera->GetRay(mousePos[0], mousePos[1], true, Xi);
	ObjectIntersection intersection = Intersect(ray);
	if (!intersection.hit)
	{
		printf("No Hit\n");
		return;
	}

	btVector3 color = intersection.material.GetColor();

	if (intersection.material.GetType() == EMIT)
	{
		printf("Emit\n");
		return;
	}

	btVector3 pos = ray.origin + ray.direction * intersection.u;
	Ray reflected = intersection.material.GetReflectedRay(ray, pos, intersection.normal, Xi);
	ObjectIntersection intersection2 = Intersect(reflected);
	btVector3 pos2 = reflected.origin + reflected.direction * intersection2.u;
	glPushMatrix();

	btTransform transform = btTransform::getIdentity();
	transform.setOrigin(pos);
	btScalar trans[16];
	transform.getOpenGLMatrix(trans);	
	glMultMatrixf(trans);

	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);

	glVertex3f(0,0,0);
	glVertex3fv(intersection.normal);

	glVertex3f(0, 0, 0);
	glVertex3fv(pos2);

	glEnd();

	DrawSphere(0.1);

	glEnable(GL_LIGHTING);
	glPopMatrix();

	printf("Hit : %f | normal :  %.1f %.1f %.1f | color : %.1f %.1f %.1f\n", intersection.u, intersection.normal[0], intersection.normal[1], intersection.normal[2], color[0], color[1], color[2]);
}

btVector3 Scene::DebugPathTest(const Ray &ray, int depth, btVector3 hitPos)
{
	ObjectIntersection intersection = Intersect(ray);
	if (!intersection.hit)
	{
		printf("0.3 0.3 0.3 -> \n");
		return btVector3(0.3, 0.3, 0.3);
	}

	btVector3 color = intersection.material.GetColor();

	if (intersection.material.GetType() == EMIT)
	{
		printf("3.3 3.3 3.3 ->\n");
		return intersection.material.GetEmission();
	}

	btVector3 pos = ray.origin + ray.direction * intersection.u;

	double maxReflection = color.x() > color.y() && color.x() > color.z() ? color.x() : color.y() > color.z() ? color.y() : color.z();
	double random = erand48();

	if (++depth > 5)
	{
		if (random < maxReflection * 0.9)
		{
			color = color * (0.9 / maxReflection);
		}
		else
		{
			return intersection.material.GetEmission();
		}
	}
	printf("%.1f %.1f %.1f -> ", color[0], color[1], color[2]);
	Ray reflected = intersection.material.GetReflectedRay(ray, pos, intersection.normal, nullptr);

	glPushMatrix();

	btTransform transform = btTransform::getIdentity();
	transform.setOrigin(pos);
	btScalar trans[16];
	transform.getOpenGLMatrix(trans);

	glDisable(GL_LIGHTING);
	glMultMatrixf(trans);
	glBegin(GL_LINES);

	glVertex3f(0, 0, 0);
	glVertex3fv(reflected.direction * 100);
	glVertex3f(0, 0, 0);
	glVertex3fv(intersection.normal);
	glEnd();

	DrawSphere(0.1);

	glEnable(GL_LIGHTING);
	glPopMatrix();

	return color * DebugPathTest(reflected, depth, pos);
}

ObjectIntersection Scene::Intersect(const Ray &ray)
{
	ObjectIntersection intersection = ObjectIntersection();
	ObjectIntersection temp;
	long size = objects.size();

	for (int i = 0; i < size; i++)
	{
		temp = objects.at((unsigned)i)->GetIntersection(ray);

		if (temp.hit)
		{
			if (intersection.u == 0 || temp.u < intersection.u)
				intersection = temp;
		}
	}

	return intersection;
}

void Scene::SaveImage(const char *filePath)
{
	int width = camera->GetWidht();
	int height = camera->GetHeight();

	std::vector<unsigned char> buffer;
	int pixelCount = width * height;

	for (int i = 0; i < pixelCount; i++)
	{
		buffer.push_back(toInt(pixelBuffer[i].x()));
		buffer.push_back(toInt(pixelBuffer[i].y()));
		buffer.push_back(toInt(pixelBuffer[i].z()));
		buffer.push_back(255);
	}

	unsigned error = lodepng::encode(filePath, buffer, width, height);
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	buffer.clear();
}

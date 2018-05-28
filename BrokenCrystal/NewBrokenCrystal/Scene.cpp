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
	for (auto& object : objects)
	{
		delete object;
	}
	objects.clear();
	objects.shrink_to_fit();
	delete camera;
	delete world;
	delete solver;
	delete broadphase;
	delete dispatcher;
	delete collisionConfiguration;
}

void Scene::Initialize()
{
	cuda_mesh_flag = 0;
	// gui 초기화
	ImGui_ImplGLUT_Init();

	// opengl Light 초기화
	GLfloat ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f }; // dark grey
	GLfloat diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // white
	GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // white
	GLfloat position[] = { 0.0f, 10.0f, 0.0f, 0.0f };

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

	//CreateBox(btVector3(0, 0, 0), btVector3(300, 1, 300), 0, Material(DIFF, btVector3(0.8, 0.8, 0.8)));

	CreateBox(btVector3(0, 0, 0), btVector3(30, 1, 30), 0, Material(DIFF, btVector3(0.2, 0.3, 0.1)));
	CreateBox(btVector3(0, 30, 0), btVector3(30, 1, 30), 0, Material(EMIT, btVector3(1.0, 1.0, 1.0), btVector3(2.2, 2.2, 2.2)));
	CreateBox(btVector3(30, 15, 0), btVector3(1, 15, 30), 0, Material(DIFF, btVector3(0.0, 0.0, 0.75)));
	CreateBox(btVector3(-30, 15, 0), btVector3(1, 15, 30), 0, Material(DIFF, btVector3(0.75, 0.0, 0.0)));
	//CreateBox(btVector3(0, 15, 30), btVector3(30, 15, 1), 0, Material(DIFF, btVector3(0.8, 0.8, 0.8)));
	CreateBox(btVector3(0, 15, -30), btVector3(30, 15, 1), 0, Material(GLOSS));
	CreateMesh(btVector3(0, 0, 30), "board.obj", 0);

	//CreateSphere(btVector3(0, 3, 0), 7, 1, Material(TRANS, btVector3(1.0, 1.0, 1.0)));

	//CreateSphere(btVector3(10, 10, 0), 2, 1, Material(SPEC, btVector3(1.0, 1.0, 1.0)));
	//CreateSphere(btVector3(0, 4, 0), 2, 1, Material(DIFF, btVector3(0.3, 0.3, 0.1)));
	//CreateSphere(btVector3(0, 10, 10), 2, 1, Material(SPEC, btVector3(1.0, 1.0, 1.0)));
	//CreateSphere(btVector3(-3, 4, 4), 4, 1, Material(DIFF, btVector3(0.3, 0.1, 0.3)));

	//CreateBox(btVector3(0, 3, 3), btVector3(2, 2, 2), 1, Material(DIFF, btVector3(0.1, 0.2, 0.1)));
	//CreateBox(btVector3(0, 2, -4), btVector3(2, 2, 2), 1, Material(SPEC, btVector3(1.0, 1.0, 1.0)));
	//CreateBox(btVector3(2, 4, 0), btVector3(2, 2, 2), 0, Material(DIFF, btVector3(0.4, 0.3, 0.1)));


	// material test
	//CreateBox(btVector3(0, 0, 0), btVector3(30, 1, 30), 0, Material(DIFF, btVector3(0.8, 0.8, 0.8)));
	//CreateBox(btVector3(0, 30, 0), btVector3(30, 1, 30), 0, Material(EMIT, btVector3(1.0, 1.0, 1.0), btVector3(2.2, 2.2, 2.2)));
	//CreateBox(btVector3(30, 15, 0), btVector3(1, 15, 30), 0, Material(DIFF, btVector3(0.0, 0.0, 0.85)));
	//CreateBox(btVector3(-30, 15, 0), btVector3(1, 15, 30), 0, Material(DIFF, btVector3(0.85, 0.0, 0.0)));
	//CreateMesh(btVector3(0, 0, 30), "board.obj", 0, Material(DIFF, btVector3(0.3, 0.5, 0.4)));
	//CreateBox(btVector3(0, 15, -30), btVector3(30, 15, 1), 0, Material(DIFF, btVector3(0.8, 0.8, 0.8)));
	//CreateSphere(btVector3(-7, 1, 3), 3, 0.1, Material(DIFF, btVector3(0.3, 0.5, 0.3)));
	//CreateSphere(btVector3(-3, 1, 3), 3, 0.1, Material(SPEC, btVector3(1.0, 1.0, 1.0)));
	//CreateSphere(btVector3(3, 1, -3), 3, 0.1, Material(GLOSS, btVector3(1.0, 1.0, 1.0)));
	//CreateSphere(btVector3(7, 1, -3), 3, 0.1, Material(TRANS, btVector3(1.0, 1.0, 1.0)));

	// island
	//CreateMesh(btVector3(0, 5, 0), "island.obj", 0, Material());
	//CreateMesh(btVector3(0, 5, 0), "water.obj", 0, Material());
	//CreateSphere(btVector3(100, 100, 100), 100, 0, Material(EMIT, btVector3(1.0, 1.0, 1.0), btVector3(6, 6, 6)));

	// dof test
	//CreateBox(btVector3(0, 0, 0), btVector3(300, 1, 300), 0, Material(DIFF, btVector3(0.8, 0.8, 0.8)));
	//CreateSphere(btVector3(0, 3, -9), 1, 0, Material(DIFF, btVector3(erand48(), erand48(), erand48())));
	//CreateSphere(btVector3(0, 3, -6), 1, 0, Material(DIFF, btVector3(erand48(), erand48(), erand48())));
	//CreateSphere(btVector3(0, 3, -3), 1, 0, Material(DIFF, btVector3(erand48(), erand48(), erand48())));
	//CreateSphere(btVector3(0, 3, 0), 1, 0, Material(DIFF, btVector3(erand48(), erand48(), erand48())));
	//CreateSphere(btVector3(0, 3, 3), 1, 0, Material(DIFF, btVector3(erand48(), erand48(), erand48())));
	//CreateSphere(btVector3(0, 3, 6), 1, 0, Material(DIFF, btVector3(erand48(), erand48(), erand48())));
	//CreateSphere(btVector3(0, 3, 9), 1, 0, Material(DIFF, btVector3(erand48(), erand48(), erand48())));

}

void Scene::AddObject(Object* object)
{
	objects.push_back(object);
	world->addRigidBody(object->GetRigidBody());
}

Mesh* Scene::CreateBox(const btVector3 &position, const btVector3 &halfExtents, float mass, Material material)
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

	Mesh* mesh = new Mesh(position, triangles, mass, material);
	AddObject(static_cast<Object*>(mesh));
	triangles.clear();
	triangles.shrink_to_fit();
	return mesh;
}

void Scene::CreateSphere(const btVector3 &position, double radius, float mass, Material material)
{
	AddObject(static_cast<Object*>(new Sphere(position, radius, mass, material)));
}

Mesh* Scene::CreateMesh(const btVector3 &position, const char* fileName, float mass, Material material)
{
	Mesh* mesh = new Mesh(position, fileName, mass, material);
	AddObject(static_cast<Object*>(mesh));
	return mesh;
}

Mesh* Scene::CreateMesh(const btVector3 &position, const char* fileName, float mass)
{
	Mesh* mesh = new Mesh(position, fileName, mass);
	AddObject(static_cast<Object*>(mesh));
	return mesh;
}

void Scene::DeleteObject(Object* object)
{
	objects.erase(std::remove(objects.begin(), objects.end(), object), objects.end());
	world->removeRigidBody(object->GetRigidBody());
	delete object;
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

	ImGuiIO& io = ImGui::GetIO();
	io.AddInputCharacter(key);
	glutPostRedisplay();
}

void Scene::KeyboardUp(unsigned char key, int x, int y)
{
	keyState[key] = false;
	mousePos[0] = x;
	mousePos[1] = y;

	if (key == 9)
		showDebugPanel = !showDebugPanel;
	glutPostRedisplay();
}

void Scene::Special(int key, int x, int y)
{
	ImGuiIO& io = ImGui::GetIO();
	io.AddInputCharacter(key);
	glutPostRedisplay();
}

void Scene::SpecialUp(int key, int x, int y)
{
	glutPostRedisplay();
}

void Scene::Reshape(int w, int h)
{
	camera->UpdateScreen(w, h);
}

void Scene::Mouse(int button, int state, int x, int y)
{
	mouseState[button] = state;
	if (IsMouseDown(2))
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

	// gui
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2(float(x), float(y));

	if (state == GLUT_DOWN && (button == GLUT_LEFT_BUTTON))
		io.MouseDown[0] = true;
	else
		io.MouseDown[0] = false;

	if (state == GLUT_DOWN && (button == GLUT_RIGHT_BUTTON))
		io.MouseDown[1] = true;
	else
		io.MouseDown[1] = false;
	glutPostRedisplay();
}

void Scene::PassiveMotion(int x, int y)
{
	mousePos[0] = x;
	mousePos[1] = y;
	deltaDrag[0] = 0;
	deltaDrag[1] = 0;
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2(float(x), float(y));
	glutPostRedisplay();
}

void Scene::Motion(int x, int y)
{
	mousePos[0] = x;
	mousePos[1] = y;
	deltaDrag[0] = 0;
	deltaDrag[1] = 0;
	isMouseDrag = true;
	if (IsMouseDown(2))
	{
		deltaDrag[0] = (clickPos[0] - x);
		deltaDrag[1] = (clickPos[1] - y);
		camera->Rotate(deltaDrag[1] * 0.1, deltaDrag[0] * 0.1);
		glutWarpPointer(clickPos[0], clickPos[1]);
		isMouseDrag = true;
	}
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2(float(x), float(y));
	glutPostRedisplay();
}

void Scene::Display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	camera->UpdateCamera();
	RenderScene();
	RenderGUI();

	glutSwapBuffers();
}


void Scene::Idle()
{
	float dt = clock.getTimeMilliseconds();
	clock.reset();
	if (isAnimation)
	{
		UpdateScene(0.06 * timeScale);
		CudaAnimationRendering(++animationFrame);
		animationTime += 0.06;
		if (animationIndex == 1 && animationTime >= 11)
		{
			ACrystalExplosion();
		}
		if (animationIndex == 2 && animationTime >= 16.5)
		{
			AStopCrystal();
		}
		if (animationIndex == 3 && animationTime >= 23.87)
		{
			AMeshExplosion(0);
		}
		if (animationIndex == 4 && animationTime >= 31.24)
		{
			AMeshExplosion(1);
		}
		if (animationIndex == 5 && animationTime >= 38.61)
		{
			AMeshExplosion(2);
		}
		if (animationIndex == 6 && animationTime >= 45.98)
		{
			AMeshExplosion(3);
		}
		if (animationIndex == 7 && animationTime >= 50)
		{
			AFinishAnimation();
		}
		if (animationTime >= 60)
		{
			isAnimation = false;
		}
	}
	else
		UpdateScene((dt / 1000.0f) * timeScale);
	glutPostRedisplay();
}

void Scene::UpdateScene(float dt)
{
	if (IsKeyDown('w'))
	{
		camera->Zoom(.5);
	}
	if (IsKeyDown('s'))
	{
		camera->Zoom(-.5);
	}
	if (IsKeyDown('q'))
	{
		camera->Fov(1);
	}
	if (IsKeyDown('e'))
	{
		camera->Fov(-1);
	}
	if (IsKeyDown('r'))
	{
		RenderPath(samples);
		SaveImage("Render.png");
		delete pixelBuffer;
	}
	if (IsKeyDown('m'))
	{
		CUMemInitialize();
	}
	if (IsKeyDown('n'))
	{
		DebugPathCU();
	}
	if (IsKeyDown('b'))
	{
		cuda_mesh_flag = 1;
	}
	if (IsKeyDown('v'))
	{
		cuda_mesh_flag = 0;
	}
	if (IsKeyDown('h'))
	{
		cudaError_t error = cudaDeviceReset();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
		}
	}
	if (IsKeyDown('d'))
	{
		DebugTraceRay();
	}
	if (IsKeyDown('f'))
	{
		Ray ray = camera->GetRay(mousePos[0], mousePos[1], 0, 0, false);
		system("cls");
		btVector3 color = DebugPathTest(ray, 0, ray.origin);
		printf("\nresult = %.1f %.1f %.1f\n", color[0], color[1], color[2]);
	}
	if (IsKeyDown('g'))
	{
		DebugTraceRay(true);
	}
	if (IsKeyDown('p'))
	{
		RenderContinuousPath();
	}


	// animation
	if (cameraRotate)
	{
		camera->Rotate(0, 1);
	}

	if (crystalExplosion)
	{
		camera->Zoom(-0.1);
		camera->Target(0, -0.01, 0);
	}


	world->stepSimulation(dt);
}

void Scene::SetTimeScale(float value)
{
	timeScale = value;
}

void Scene::RenderGUI()
{
	int width = camera->GetWidht();
	int height = camera->GetHeight();
	ImGui_ImplGLUT_NewFrame(width, height);

	if (showDebugPanel)
	{
		ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiSetCond_Once);
		ImGui::SetNextWindowSize(ImVec2(width / 3, height / 2), ImGuiSetCond_Appearing);
		ImGui::Begin("Debug Panel", &showDebugPanel);
		if(isTracing)
			ImGui::Text("Raytracing... %0.1f%% [ETC %.3dh%.2dm%.2ds]", completion, remaining / 3600, (remaining % 3600) / 60, remaining % 60);
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::InputInt("Samples", &samples, 1, 1);
		samples = dmax(samples, 1);
		ImGui::SliderFloat("Fov", &camera->GetFovPointer(), 1, 179);
		ImGui::SliderFloat("Zoom", &camera->GetDistancePointer(), 0, 20);
		ImGui::SliderFloat3("Position", camera->GetPositionPointer(), -100, 100);
		ImGui::SliderFloat3("Target", camera->GetTargetPointer(), -100, 100);
		ImGui::SliderFloat("Pitch", &camera->GetPitchPointer(), 1, 89);
		ImGui::SliderFloat("Yaw", &camera->GetYawPointer(), -360, 360);
		if (!isTracing && ImGui::Button("Start Tracing"))
		{
			RenderPath(samples);
			SaveImage("Render.png");
		}
		if (ImGui::Button("Start Animation"))
		{
			Animation();
		}
		ImGui::End();
	}

	ImGui::Render();
}

void Scene::RenderScene()
{
	if (cuda_mesh_flag == 0)
	{
		for (auto & object : objects)
		{
			DrawShape(object);
		}
	}
	else
	{
		DrawMeshDebugCU();
	}
}

// CUDA

void Scene::CUMemInitialize()
{
	size_t freemem;
	size_t totalmem;

	cudaError_t error = cudaMemGetInfo(&freemem, &totalmem);
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	std::cout << &freemem << " " << &totalmem << std::endl;
	// Camera
	CameraCU* cam_cuda = new CameraCU;
	camera->CopyCamera(cam_cuda);
	CameraCU* cam_p;
	
	cudaMalloc((void**)&cam_p, sizeof(CameraCU));	//cam_p contains device memory address
	cudaMemcpy(cam_p, cam_cuda, sizeof(CameraCU), cudaMemcpyHostToDevice);

	std::cout << "cam copy to device successed" << std::endl;
	// Objects
	std::vector<ObjectCU*>* loaded_object = new std::vector<ObjectCU*>;	//loaded_object contains array of device memory address of Object
	for (auto & object : objects)
	{
		loaded_object->push_back(CULoadObj(object));
	}

	ObjectCU** objects_p;
	cudaMalloc((void**)&objects_p, loaded_object->size() * sizeof(ObjectCU**));
	cudaMemcpy(objects_p, loaded_object->data(), loaded_object->size() * sizeof(ObjectCU**), cudaMemcpyHostToDevice);
	std::cout << "objects copy to device successed" << std::endl;

	// Objects num
	int* num_objects_device;
	int num_objects_host = loaded_object->size();
	cudaMalloc((void**)&num_objects_device, sizeof(int));
	cudaMemcpy(num_objects_device, &num_objects_host, sizeof(int), cudaMemcpyHostToDevice);

	TracePath* tp = new TracePath;


	float3* result = tp->RenderPathCU(objects_p, num_objects_device, cam_p, camera->GetWidht(), camera->GetHeight());
	std::cout << "CU path tracing finished! Start SaveImageCU.." << std::endl;
	SaveImageCU(result, "RenderCU.png");
	delete tp;

	std::cout << "SaveImageCU successed!" << std::endl;
	// need to pass cam_p, objects_p, objects_p size
}

ObjectCU* Scene::CULoadObj(Object* object)
{
	float transform[16];
	btCollisionShape* pShape = object->GetShape();
	object->GetTransform(transform);

	switch (pShape->getShapeType())
	{
		/*
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
		}*/

		case CONVEX_TRIANGLEMESH_SHAPE_PROXYTYPE:
		{
			const Mesh* mesh = static_cast<const Mesh*>(object);
			Triangle* t;
			std::vector<float3>* triangles = new std::vector<float3>;
			for (int i = 0; i < mesh->GetTriangles().size(); i++)
			{
				t = mesh->GetTriangles().at((unsigned)i);
				triangles->push_back(make_float3(t->pos[0].getX(), t->pos[0].getY(), t->pos[0].getZ()) * transform);
				triangles->push_back(make_float3(t->pos[1].getX(), t->pos[1].getY(), t->pos[1].getZ()) * transform);
				triangles->push_back(make_float3(t->pos[2].getX(), t->pos[2].getY(), t->pos[2].getZ()) * transform);
			}
			float3* mesh_p;
			unsigned int triangles_size = (unsigned int)(triangles->size() * sizeof(float3));
			cudaMalloc((void**)&mesh_p, triangles_size);	//mesh_p contains device memory address for triangles
			cudaMemcpy(mesh_p, triangles->data(), triangles_size, cudaMemcpyHostToDevice);

			ObjectCU* temp = new ObjectCU;
			temp->triangles_size = triangles_size;
			temp->triangles_num = (unsigned int)triangles->size();
			temp->triangles_p = mesh_p;
			temp->material = mesh->GetTriangles().at((unsigned)0)->GetMaterial().GetType();
			temp->color = mesh->GetTriangles().at((unsigned)0)->GetMaterial().GetColorF();
			temp->emission = mesh->GetTriangles().at((unsigned)0)->GetMaterial().GetEmissionF();;

			ObjectCU* object_p;
			cudaMalloc((void**)&object_p, sizeof(ObjectCU));
			cudaMemcpy(object_p, temp, sizeof(ObjectCU), cudaMemcpyHostToDevice);

			return object_p;
		}		
	}
}

void Scene::DrawMeshDebugCU()
{
	std::vector<ObjectCU*> loaded_object;	//loaded_object contains array of device memory address of Object
	int testval = 0;
	for (auto & object : objects)
	{
		float transform[16];
		object->GetTransform(transform);

		const Mesh* mesh = static_cast<const Mesh*>(object);
		Triangle* t;
		std::vector<float3>* triangles = new std::vector<float3>;
		for (int i = 0; i < mesh->GetTriangles().size(); i++)
		{
			t = mesh->GetTriangles().at((unsigned)i);
			triangles->push_back(make_float3(t->pos[0].getX(), t->pos[0].getY(), t->pos[0].getZ()) * transform);
			triangles->push_back(make_float3(t->pos[1].getX(), t->pos[1].getY(), t->pos[1].getZ()) * transform);
			triangles->push_back(make_float3(t->pos[2].getX(), t->pos[2].getY(), t->pos[2].getZ()) * transform);
		}


		unsigned int triangles_size = (unsigned int)(triangles->size() * sizeof(float3));

		ObjectCU* temp = new ObjectCU;
		temp->triangles_size = triangles_size;
		temp->triangles_num = (unsigned int)triangles->size();
		temp->triangles_p = triangles->data();
		temp->material = object->GetMaterial().GetType();
		temp->color = object->GetMaterial().GetColorF();
		temp->emission = object->GetMaterial().GetEmissionF();

		loaded_object.push_back(temp);
		std::cout << temp->triangles_p << std::endl;
	}

	for (int i = 0; i < loaded_object.size(); i++)
	{
		ObjectCU* mesh = loaded_object[i];
		
		glColor3f(mesh->color.x, mesh->color.y, mesh->color.z);

		for (int i = 0; i < mesh->triangles_num; i += 3)
		{
			btVector3 p0 = btVector3(mesh->triangles_p[i].x, mesh->triangles_p[i].y, mesh->triangles_p[i].z);
			btVector3 p1 = btVector3(mesh->triangles_p[i + 1].x, mesh->triangles_p[i + 1].y, mesh->triangles_p[i + 1].z);
			btVector3 p2 = btVector3(mesh->triangles_p[i + 2].x, mesh->triangles_p[i + 2].y, mesh->triangles_p[i + 2].z);
			DrawTriangle(p0, p1, p2);
		}
	}
}

void Scene::CudaAnimationRendering(int index)
{
	size_t freemem;
	size_t totalmem;

	cudaError_t error = cudaMemGetInfo(&freemem, &totalmem);
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	std::cout << &freemem << " " << &totalmem << std::endl;
	// Camera
	CameraCU* cam_cuda = new CameraCU;
	camera->CopyCamera(cam_cuda);
	CameraCU* cam_p;

	cudaMalloc((void**) &cam_p, sizeof(CameraCU));	//cam_p contains device memory address
	cudaMemcpy(cam_p, cam_cuda, sizeof(CameraCU), cudaMemcpyHostToDevice);

	std::cout << "cam copy to device successed" << std::endl;
	// Objects
	std::vector<ObjectCU*>* loaded_object = new std::vector<ObjectCU*>;	//loaded_object contains array of device memory address of Object
	for (auto & object : objects)
	{
		loaded_object->push_back(CULoadObj(object));
	}

	ObjectCU** objects_p;
	cudaMalloc((void**) &objects_p, loaded_object->size() * sizeof(ObjectCU**));
	cudaMemcpy(objects_p, loaded_object->data(), loaded_object->size() * sizeof(ObjectCU**), cudaMemcpyHostToDevice);
	std::cout << "objects copy to device successed" << std::endl;

	// Objects num
	int* num_objects_device;
	int num_objects_host = loaded_object->size();
	cudaMalloc((void**) &num_objects_device, sizeof(int));
	cudaMemcpy(num_objects_device, &num_objects_host, sizeof(int), cudaMemcpyHostToDevice);

	TracePath* tp = new TracePath;

	float3* result = tp->RenderPathCU(objects_p, num_objects_device, cam_p, camera->GetWidht(), camera->GetHeight());
	std::cout << "CU path tracing finished! Start SaveImageCU.." << std::endl;

	std::string fileName = "Animation/";
	fileName.append(std::to_string(index));
	fileName.append(".png");
	SaveImageCU(result, fileName.c_str());
	delete tp;
	std::cout << "SaveImageCU successed!" << std::endl;
}

void Scene::DebugPathCU()
{
	// Camera
	CameraCU* cam_cuda = new CameraCU;
	camera->CopyCamera(cam_cuda);
	CameraCU* cam_p;

	cudaMalloc((void**)&cam_p, sizeof(CameraCU));	//cam_p contains device memory address
	cudaMemcpy(cam_p, cam_cuda, sizeof(CameraCU), cudaMemcpyHostToDevice);

	//std::cout << "cam copy to device successed" << std::endl;
	// Objects
	std::vector<ObjectCU*> *loaded_object = new std::vector<ObjectCU*>;	//loaded_object contains array of device memory address of Object
	for (auto & object : objects)
	{
		loaded_object->push_back(CULoadObj(object));
	}

	ObjectCU** objects_p;
	cudaMalloc((void**)&objects_p, loaded_object->size() * sizeof(ObjectCU**));
	cudaMemcpy(objects_p, loaded_object->data(), loaded_object->size() * sizeof(ObjectCU**), cudaMemcpyHostToDevice);
	//std::cout << "objects copy to device successed" << std::endl;

	TracePath* tp = new TracePath;


	tp->RenderPathCUDebug(objects_p, loaded_object->size(), cam_p, mousePos);
	//std::cout << "RenderPathCUDebug successed" << std::endl;
	delete tp;
	delete loaded_object;
	delete cam_cuda;

	// need to pass cam_p, objects_p, objects_p size
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
			DrawMesh(mesh);
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

void Scene::DrawMesh(const Mesh* mesh)
{
	std::vector<Triangle*> triangles = mesh->GetTriangles();
	for (auto & triangle : triangles)
	{
		glColor3fv(triangle->GetColorAt());
		DrawTriangle(triangle->pos[0], triangle->pos[1], triangle->pos[2]);
	}
}

void Scene::RenderPath(int samples)
{
	isTracing = true;
	int width = camera->GetWidht();
	int height = camera->GetHeight();
	double samplesP = 1.0 / samples;
	pixelBuffer = new btVector3[width * height];
	remaining = 0;
	completion = 0;
	unsigned int startTime = time(nullptr);
	unsigned int lastTime = startTime;
	unsigned int progress = 0;
	unsigned int lastProgress = 0;
	float lastSpeed = 0;
	int pixelCount = width * height;
#pragma omp parallel for schedule(dynamic, 1)
	
	for (int t = 0; t < pixelCount; t++)
	{
		int x = t % width;
		int y = t / width;
		btVector3 resultColor = btVector3(0, 0, 0);

		for (int sy = 0; sy < 2; sy++)
		{
			for (int sx = 0; sx < 2; sx++)
			{
				btVector3 color = btVector3(0, 0, 0);
				for (int s = 0; s < samples; s++)
				{
					//Ray ray = camera->GetRay(x, y, s > 0);
					Ray ray = camera->GetRay(x, y, sx, sy, false); // dof 효과 미완성
					color = color + TraceRay(ray, 0);
					//printf("%f %f %f\n", color[0], color[1], color[2]);
					//Sleep(1000);
				}
				resultColor = resultColor + color * samplesP;
				//printf("final : %f %f %f\n", color[0] * samplesP, color[1] * samplesP, color[2] * samplesP);
			}
		}
		pixelBuffer[(y)* width + x] = resultColor * 0.25;

#pragma omp critical
		{
			++progress;
			if (progress > 0 && ((float)difftime(time(nullptr), lastTime) >= 1.0f))
			{
				completion = (float)(progress - 1) / pixelCount * 100;
				float speed = (float)(progress - lastProgress) / difftime(time(nullptr), lastTime);
				remaining = (int)((pixelCount - progress) / speed);
				printf("\rPathTracing (%d samples)  %0.1f%% [ETC %.3dh%.2dm%.2ds]", samples, completion, remaining / 3600, (remaining % 3600) / 60, remaining % 60);
				lastTime = time(nullptr);
				lastProgress = progress;
				lastSpeed = speed;
			}
		}
	}
	int elapsedTime = (int)difftime(time(nullptr), startTime);
	printf("\rPathTracing complete, time taken: %.2dh%.2dm%.2ds.\n", elapsedTime / 3600, (elapsedTime % 3600) / 60, elapsedTime % 60);
	isTracing = false;
}

void Scene::RenderContinuousPath(int maxSamples)
{
	isTracing = true;
	int width = camera->GetWidht();
	int height = camera->GetHeight();
	pixelBuffer = new btVector3[width * height];
	for (int i = 1; i < maxSamples; i++)
	{
		remaining = 0;
		completion = 0;
		unsigned int startTime = time(nullptr);
		unsigned int lastTime = startTime;
		unsigned int progress = 0;
		unsigned int lastProgress = 0;
		float lastSpeed = 0;
		int pixelCount = width * height;
#pragma omp parallel for schedule(dynamic, 1)
		for (int t = 0; t < pixelCount; t++)
		{
			int x = t % width;
			int y = t / width;
			btVector3 resultColor = btVector3(0, 0, 0);

			for (int sy = 0; sy < 2; sy++)
			{
				for (int sx = 0; sx < 2; sx++)
				{
					Ray ray = camera->GetRay(x, y, sx, sy, false); // dof 효과 미완성
					resultColor += TraceRay(ray, 0);
				}
			}
			btVector3 post = pixelBuffer[(y) * width + x];
			pixelBuffer[(y) * width + x] = (post * (i - 1) + resultColor * 0.25) / i;
#pragma omp critical
			{
				++progress;
				if (progress > 0 && ((float) difftime(time(nullptr), lastTime) >= 1.0f))
				{
					completion = (float) (progress - 1) / pixelCount * 100;
					float speed = (float) (progress - lastProgress) / difftime(time(nullptr), lastTime);
					remaining = (int) ((pixelCount - progress) / speed);
					printf("\rPathTracing (%d)  %0.1f%% [ETC %.3dh%.2dm%.2ds]", i, completion, remaining / 3600, (remaining % 3600) / 60, remaining % 60);
					lastTime = time(nullptr);
					lastProgress = progress;
					lastSpeed = speed;
				}
			}
		}
		std::string fileName;
		fileName.append("Render");
		fileName.append(std::to_string(i));
		fileName.append(".png");
		SaveImage(fileName.c_str());
	}
	isTracing = false;
}


btVector3 Scene::TraceRay(const Ray &ray, int depth)
{
	ObjectIntersection intersection = Intersect(ray);
	if (!intersection.hit) return btVector3(0.0, 0.0, 0.0);
	if (intersection.material.GetType() == EMIT)
		return intersection.material.GetEmission();

	btVector3 color = intersection.material.GetColor();
	double maxReflection = color.x()>color.y() && color.x()>color.z() ? color.x() : color.y()>color.z() ? color.y() : color.z();
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

	btVector3 pos = ray.origin + ray.direction * intersection.u;
	Ray reflected = intersection.material.GetReflectedRay(ray, pos, intersection.normal, color);
	return color * TraceRay(reflected, depth);
}

void Scene::DebugTraceRay(bool dof)
{
	Ray ray = camera->GetRay(mousePos[0], mousePos[1], 0, 0, dof);
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
	Ray reflected = intersection.material.GetReflectedRay(ray, pos, intersection.normal, color);
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
	Ray reflected = intersection.material.GetReflectedRay(ray, pos, intersection.normal, color);

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

	int pixelCount = width * height;
	std::string fileName;
	std::vector<unsigned char> buffer;
	unsigned char * out = nullptr;

	for (int i = 0; i < pixelCount; i++)
	{
		buffer.push_back(toInt(pixelBuffer[i].x()));
		buffer.push_back(toInt(pixelBuffer[i].y()));
		buffer.push_back(toInt(pixelBuffer[i].z()));
	}
	fileName.append("before");
	fileName.append(filePath);
	unsigned error = lodepng::encode(fileName.c_str(), buffer, width, height, LCT_RGB);
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	for (int i = 0; i < 100; i++)
		recursive_bf(buffer.data(), out, 0.01, 0.5, width, height, 3);

	fileName = std::string("after");
	fileName.append(filePath);
	error = lodepng::encode(fileName.c_str(), out, width, height, LCT_RGB);
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	std::vector<unsigned char> vclear;
	buffer.swap(vclear);
	vclear.clear();
	buffer.clear();
	buffer.shrink_to_fit();	
}

void Scene::SaveImageCU(float3* pixels, const char *filePath)
{
	int width = camera->GetWidht();
	int height = camera->GetHeight();

	std::vector<unsigned char> buffer;
	int pixelCount = width * height;

	for (int i = 0; i < pixelCount; i++)
	{
		buffer.push_back(toInt(pixels[i].x));
		buffer.push_back(toInt(pixels[i].y));
		buffer.push_back(toInt(pixels[i].z));
		buffer.push_back(255);

		//printf("%d %d %d\n", toInt(pixels[i].x), toInt(pixels[i].y), toInt(pixels[i].z));
	}

	unsigned error = lodepng::encode(filePath, buffer, width, height);
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	std::vector<unsigned char> vclear;
	buffer.swap(vclear);
	vclear.clear();
	buffer.clear();
	buffer.shrink_to_fit();
	delete pixels;
}

void Scene::Animation()
{
	// Init
	currentCrystal = CreateMesh(btVector3(0, 15, 0), "Crystal_Low.obj", 0, Material(GLOSS, btVector3(0.4, 0.4, 1.0)));
	camera->SetTarget(currentCrystal->GetPosition());
	camera->SetPitch(31.5f);
	camera->SetYaw(-180);
	camera->SetZoom(6);

	currentMeshes.push_back(CreateMesh(btVector3(-7, 7, 5), "corn.obj", 0.1, Material(DIFF, btVector3(0.3, 0.5, 0.3))));
	currentMeshes.push_back(CreateMesh(btVector3(-3, 7, 5), "corn.obj", 0.1, Material(SPEC, btVector3(1.0, 1.0, 1.0))));
	currentMeshes.push_back(CreateMesh(btVector3(3, 7, -5), "corn.obj", 0.1, Material(GLOSS, btVector3(1.0, 1.0, 1.0))));
	currentMeshes.push_back(CreateMesh(btVector3(7, 7, -5), "corn.obj", 0.1, Material(TRANS, btVector3(1.0, 1.0, 1.0))));
	// start
	ARotateCamera();
}

void Scene::ARotateCamera()
{
	printf("[Animation] Rotate Camera\n");
	animationIndex = 1;
	isAnimation = true;
	cameraRotate = true;
	//glutTimerFunc(3000, &Scene::ACrystalExplosion, 0);
}

void Scene::ACrystalExplosion()
{
	printf("[Animation] Crystal Explosion\n");
	animationIndex = 2;
	cameraRotate = false;
	crystalExplosion = true;
	btVector3 crystalPos = currentCrystal->GetPosition();
	crystalPos[1] -= 2;
	DeleteObject(currentCrystal);
	currentCrystal = CreateMesh(btVector3(0, 15, 0), "Crystal_Low.obj", 10, Material(GLOSS, btVector3(0.4, 0.4, 1.0)));
	std::vector<Mesh*> meshes = break_into_pieces2(currentCrystal, 50);
	for (auto& mesh : meshes)
	{
		AddObject(static_cast<Object*>(mesh));
		btVector3 meshPos = mesh->GetPosition();
		btVector3 dir = (meshPos - crystalPos).normalize();
		mesh->GetRigidBody()->applyCentralImpulse(dir / 10.0f);
	}
	DeleteObject(currentCrystal);
}

void Scene::AStopCrystal()
{
	printf("[Animation] Freeze Crystal\n");
	animationIndex = 3;
	crystalExplosion = false;
	cameraRotate = true;
	SetTimeScale(0.05f);
}

void Scene::AMeshExplosion(int index)
{
	Mesh* currentMesh = currentMeshes.at(index);
	animationIndex++;
	btVector3 currentMeshPos = currentMesh->GetPosition();
	std::vector<Mesh*> meshes = break_into_pieces2(currentMesh, 20);
	for (auto& mesh : meshes)
	{
		AddObject(static_cast<Object*>(mesh));
		btVector3 meshPos = mesh->GetPosition();
		btVector3 dir = (meshPos - currentMeshPos).normalize();
		mesh->GetRigidBody()->applyCentralImpulse(dir / 30.0f);
	}
	DeleteObject(currentMesh);
}

void Scene::AFinishAnimation()
{
	printf("[Animation] Animation Finished\n");
	animationIndex = 8;
	cameraRotate = false;
	SetTimeScale(1.0f);
}


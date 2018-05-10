#include "Scene.h"

Scene::Scene()
{
	camera = Camera::Create();
	cube = Object::Create();
	InputManager::OnMouseDrag.permanent_bind([this](float deltaX, float deltaY) { this->camera->Rotate(deltaY * 0.1f, deltaX * 0.1f, 0); });
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
	if (InputManager::IsKeyDown('t'))
	{
		cube->LookAt(camera->position);
	}
}

void Scene::Render()
{
	glMultMatrixf(value_ptr(camera->GetViewMatrix()));
	DrawGrid(50, 5);
	DrawDebugCube();
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

void Scene::DrawDebugCube()
{
	glPushMatrix();

	glMultMatrixf(value_ptr(cube->GetViewMatrix()));
	DrawAxis(10);
	glBegin(GL_QUADS);
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(1.0, 1.0, 1.0);
	glVertex3f(-1.0, 1.0, 1.0);
	glVertex3f(-1.0, 1.0, -1.0);
	glVertex3f(1.0, 1.0, -1.0);

	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(1.0, -1.0, 1.0);
	glVertex3f(-1.0, -1.0, 1.0);
	glVertex3f(-1.0, -1.0, -1.0);
	glVertex3f(1.0, -1.0, -1.0);

	glColor3f(1.0, 1.0, 0.0);
	glVertex3f(1.0, 1.0, -1.0);
	glVertex3f(-1.0, 1.0, -1.0);
	glVertex3f(-1.0, -1.0, -1.0);
	glVertex3f(1.0, -1.0, -1.0);

	glColor3f(0.0, 0.0, 1.0);
	glVertex3f(1.0, 1.0, 1.0);
	glVertex3f(1.0, 1.0, -1.0);
	glVertex3f(1.0, -1.0, -1.0);
	glVertex3f(1.0, -1.0, 1.0);

	glColor3f(1.0, 0.5, 0.0);
	glVertex3f(-1.0, 1.0, 1.0);
	glVertex3f(-1.0, 1.0, -1.0);
	glVertex3f(-1.0, -1.0, -1.0);
	glVertex3f(-1.0, -1.0, 1.0);

	glColor3f(1.0, 0.5, 1.0);
	glVertex3f(1.0, 1.0, 1.0);
	glVertex3f(-1.0, 1.0, 1.0);
	glVertex3f(-1.0, -1.0, 1.0);
	glVertex3f(1.0, -1.0, 1.0);
	glEnd();
	glPopMatrix();
}

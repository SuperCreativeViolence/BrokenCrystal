#include "Init.h"
#include "Scene.h"

Scene::p scene = Scene::Create();

void Mouse(int mouse_event, int state, int x, int y)
{
	InputManager::MouseInput(mouse_event, state, x, y);

	glutPostRedisplay();
}

void Motion(int x, int y)
{
	InputManager::MouseMotion(x, y);
	glutPostRedisplay();
}

void Rendering(void)
{
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();

	//scene->RenderScene();
	//
	//glutSwapBuffers();

}

void Reshape(int w, int h)
{
	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, (float) w / h, 0.1, 500);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void KeyUp(unsigned char key, int x, int y)
{
	InputManager::KeyboardInput(key, false, x, y);
}

void KeyDown(unsigned char key, int x, int y)
{
	InputManager::KeyboardInput(key, true, x, y);
}

void Timer(int timer)
{
	scene->Update();
	glutPostRedisplay();
	glutTimerFunc(1000 / 60, Timer, 1);
}

void Idle()
{
	scene->Idle();
}

void EventHandlingAndLoop()
{
	glutKeyboardFunc(KeyDown);
	glutKeyboardUpFunc(KeyUp);
	glutDisplayFunc(Rendering);
	glutReshapeFunc(Reshape);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutIdleFunc(Idle);
	glutTimerFunc(1000/60, Timer, 1);

	glutMainLoop();
}

int main(int argc, char** argv)
{
	Initialize(argc, argv);
	scene->Initialize();
	EventHandlingAndLoop();
	return 0;
}
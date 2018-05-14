#include "Init.h"
#include "Scene.h"

Scene* scene = new Scene(SCREEN_WIDTH, SCREEN_HEIGHT, 1, true);

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

}

void Reshape(int w, int h)
{
	scene->Reshape(w, h);
}

void KeyUp(unsigned char key, int x, int y)
{
	InputManager::KeyboardInput(key, false, x, y);
}

void KeyDown(unsigned char key, int x, int y)
{
	InputManager::KeyboardInput(key, true, x, y);
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
	glutPassiveMotionFunc(Motion);
	glutIdleFunc(Idle);

	glutMainLoop();
}

int main(int argc, char** argv)
{
	Initialize(argc, argv);
	scene->Initialize();
	EventHandlingAndLoop();
	return 0;
}
#ifndef GLUTCALLBACK_H
#define GLUTCALLBACK_H

#include "Scene.hpp"

static Scene* scene;

static void KeyboardCallback(unsigned char key, int x, int y)
{
	scene->Keyboard(key, x, y);
}
static void KeyboardUpCallback(unsigned char key, int x, int y)
{
	scene->KeyboardUp(key, x, y);
}
static void SpecialCallback(int key, int x, int y)
{
	scene->Special(key, x, y);
}
static void SpecialUpCallback(int key, int x, int y)
{
	scene->SpecialUp(key, x, y);
}
static void ReshapeCallback(int w, int h)
{
	scene->Reshape(w, h);
}
static void IdleCallback()
{
	scene->Idle();
}
static void MouseCallback(int button, int state, int x, int y)
{
	scene->Mouse(button, state, x, y);
}
static void MotionCallback(int x, int y)
{
	scene->Motion(x, y);
}
static void DisplayCallback(void)
{
	scene->Display();
}

int glutmain(int argc, char **argv, int width, int height, const char* title, Scene* pScene)
{
	scene = pScene;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(width, height);
	glutCreateWindow(title);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	scene->Initialize();

	glutKeyboardFunc(KeyboardCallback);
	glutKeyboardUpFunc(KeyboardUpCallback);
	glutSpecialFunc(SpecialCallback);
	glutSpecialUpFunc(SpecialUpCallback);
	glutReshapeFunc(ReshapeCallback);
	glutIdleFunc(IdleCallback);
	glutMouseFunc(MouseCallback);
	glutPassiveMotionFunc(MotionCallback);
	glutMotionFunc(MotionCallback);
	glutDisplayFunc(DisplayCallback);

	glutMainLoop();
	return 0;
}

#endif
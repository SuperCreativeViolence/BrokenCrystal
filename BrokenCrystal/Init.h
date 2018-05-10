#pragma once
#include <iostream>
#include <gl/glut.h>

#define SCREEN_HEIGHT 720
#define SCREEN_WIDTH 1280

void Initialize(int argc, char** argv)
{
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
	glutCreateWindow("SCV");
	//glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	
	glEnable(GL_DEPTH_TEST);

	glClearColor(0, 0, 0, 1.0f);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_BACK);
}
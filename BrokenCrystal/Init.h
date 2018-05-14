#pragma once
#include <iostream>
#include <gl/freeglut.h>

#define SCREEN_WIDTH 150
#define SCREEN_HEIGHT 150

void Initialize(int argc, char** argv)
{
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
	glutCreateWindow("SCV");
	
	glEnable(GL_DEPTH_TEST);

	glClearColor(0, 0, 0, 1.0f);

	glEnable(GL_CULL_FACE);
}
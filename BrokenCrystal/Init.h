#pragma once
#include <iostream>
#include <gl/glut.h>

bool key_state[256] = { false };

void Lighting(void) 
{
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

	float light_pos[] = { 25.0f, 50.0f, 50.0f, 0.0f };
	float light_dir[] = { 0.0f, -1.0f, -1.0f };
	float light_ambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	float light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	float light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };

	float matShininess = 200;
	float noMat[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	float matSpec[] = { 0.2f, 0.2f, 0.2f, 1.0f };

	glMaterialfv(GL_FRONT, GL_EMISSION, noMat);
	glMaterialfv(GL_FRONT, GL_SPECULAR, matSpec);
	glMaterialf(GL_FRONT, GL_SHININESS, matShininess);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
	glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, light_dir);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
}

void Initialize(int argc, char** argv)
{
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowPosition(400, 100);
	glutInitWindowSize(1280, 720);
	glutCreateWindow("SCV");
	
	glEnable(GL_DEPTH_TEST);

	glClearColor(0, 0, 0, 1.0f);

	glDisable(GL_CULL_FACE);

	Lighting();
}
#include "Init.h"

using namespace std;

void Mouse(int mouse_event, int state, int x, int y)
{

	glutPostRedisplay();
}

void Motion(int x, int y)
{
	
	glutPostRedisplay();
}

void Rendering(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	glutSwapBuffers();

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

void Keyboard(unsigned char key, int x, int y)
{

	glutPostRedisplay();
}

void EventHandlingAndLoop()
{
	glutKeyboardFunc(Keyboard); 
	glutDisplayFunc(Rendering); 
	glutReshapeFunc(Reshape);   
	glutMouseFunc(Mouse);       
	glutMotionFunc(Motion);      

	glutMainLoop();
}

int main(int argc, char** argv)
{
	Initialize(argc, argv);

	EventHandlingAndLoop();
	return 0;
}
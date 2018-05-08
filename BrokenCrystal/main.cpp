#include "Init.h"
#include "Camera.h"

using namespace std;

Camera* camera;
Object* cube;

void DrawAxis(int size)
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

void DrawGrid(float size, float step)
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

void DrawCube()
{
	glPushMatrix();

	glTranslatef(cube->position.x, cube->position.y, cube->position.z);
	mat4 view_matrix = cube->GetViewMatrix();
	glMultMatrixf(value_ptr(view_matrix));

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
	gluLookAt(camera->position.x, camera->position.y, camera->position.z, 0, 0, 0, 0, 1, 0);

	DrawGrid(50, 5);
	DrawCube();

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
	switch (key)
	{
		case 'w':
			cube->Rotate(vec3(10, 0, 0));
			break;
		case 's':
			cube->Rotate(vec3(-10, 0, 0));
			break;
		case 'a':
			cube->Rotate(vec3(0, 10, 0));
			break;
		case 'd':
			cube->Rotate(vec3(0, -10, 0));
			break;
		case 'q':
			cube->Rotate(vec3(0, 0, 10));
			break;
		case 'e':
			cube->Rotate(vec3(0, 0, -10));
			break;
		default:
			break;
	}
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

	camera = new Camera();
	cube = new Object();

	EventHandlingAndLoop();
	return 0;
}
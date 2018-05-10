#include "Init.h"
#include "Camera.h"

bool IsLMBPressed = false;
float DragX, DragY;
float DragPrevX, DragPrevY;
float DragDeltaX, DragDeltaY;

Camera::p camera = Camera::Create();
Object::p cube = Object::Create();

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

void SetDeltaDrag()
{
	camera->Rotate(DragDeltaY, DragDeltaX, 0);
}

void Mouse(int mouse_event, int state, int x, int y)
{
	IsLMBPressed = mouse_event == 0 && state == 0;
	if (IsLMBPressed)
	{
		DragPrevX = x;
		DragPrevY = y;
	}

	glutPostRedisplay();
}

void Motion(int x, int y)
{
	if (IsLMBPressed)
	{
		DragX = x;
		DragY = y;
		DragDeltaX = (DragPrevX - DragX) * 0.1f;
		DragDeltaY = (DragPrevY - DragY) * 0.1f;
		DragPrevX = DragX;
		DragPrevY = DragY;
		SetDeltaDrag();
	}
	glutPostRedisplay();
}

void Rendering(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMultMatrixf(value_ptr(camera->GetViewMatrix()));

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

void KeyUp(unsigned char key, int x, int y)
{
	key_state[key] = false;
}

void KeyDown(unsigned char key, int x, int y)
{
	key_state[key] = true;
}

void Timer(int timer)
{
	if (key_state['w'])
	{
		camera->Translate(0, 0, -1);
	}
	if (key_state['s'])
	{
		camera->Translate(0, 0, 1);
	}
	if (key_state['a'])
	{
		camera->Translate(-1, 0, 0);
	}
	if (key_state['d'])
	{
		camera->Translate(1, 0, 0);
	}
	if (key_state['q'])
	{
		camera->Translate(0, -1, 0);
	}
	if (key_state['e'])
	{
		camera->Translate(0, 1, 0);
	}

	if (key_state['t'])
	{
		cube->LookAt(camera->position);
	}

	glutPostRedisplay();
	glutTimerFunc(1000 / 60, Timer, 1);
}

void EventHandlingAndLoop()
{
	glutKeyboardFunc(KeyDown);
	glutKeyboardUpFunc(KeyUp);
	glutDisplayFunc(Rendering);
	glutReshapeFunc(Reshape);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutTimerFunc(1000/60, Timer, 1);

	glutMainLoop();
}

int main(int argc, char** argv)
{
	Initialize(argc, argv);
	EventHandlingAndLoop();
	return 0;
}
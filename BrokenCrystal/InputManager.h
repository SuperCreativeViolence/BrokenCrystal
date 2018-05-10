#ifndef INPUTMANAGER_H
#define INPUTMANAGER_H

#include "Event.h"
#include "gl/glut.h"

class InputManager
{
public:
	static void KeyboardInput(unsigned char key, bool down, int x, int y);
	static void MouseInput(int mouse_event, int state, int x, int y);
	static void MouseMotion(int x, int y);
	static int* GetDragDelta();
	static bool IsKeyDown(unsigned char key);
	static bool IsMouseDown(int mouse);

	static Event<int, int> OnMouseDrag;
	static Event<int, int> OnMouseMove;
	static Event<int, int, int, int> OnMouseClick;

private:
	static bool key_state[256];
	static bool mouse_state[3];
	static int click_pos[2];
	static int drag_delta[2];
};

#endif
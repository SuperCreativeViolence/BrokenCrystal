#ifndef INPUTMANAGER_H
#define INPUTMANAGER_H
#include <cmath>

class InputManager
{
public:
	static void KeyboardInput(unsigned char key, bool down, int x, int y);
	static void MouseInput(int mouse_event, int state, int x, int y);
	static void MouseMotion(int x, int y);
	static float* GetDragDelta();
	static bool IsKeyDown(unsigned char key);
	static bool IsMouseDown(int mouse);

private:
	static bool key_state[256];
	static bool mouse_state[3];
	static float drag_prev[2];
	static float drag_delta[2];
};

#endif
#include "InputManager.h"
#include <stdio.h>

bool InputManager::key_state[256];
bool InputManager::mouse_state[3];
float InputManager::drag_prev[2];
float InputManager::drag_delta[2];
Event<float, float> InputManager::OnMouseDrag;

void InputManager::KeyboardInput(unsigned char key, bool down, int x, int y)
{
	key_state[key] = down;
}

void InputManager::MouseInput(int mouse_event, int state, int x, int y)
{
	mouse_state[mouse_event] = state;
	if (IsMouseDown(0))
	{
		drag_prev[0] = x;
		drag_prev[1] = y;
	}
}

void InputManager::MouseMotion(int x, int y)
{
	if (IsMouseDown(0))
	{
		drag_delta[0] = (drag_prev[0] - x);
		drag_delta[1] = (drag_prev[1] - y);
		drag_prev[0] = x;
		drag_prev[1] = y;
		OnMouseDrag.fire(drag_delta[0], drag_delta[1]);
	}
}

float* InputManager::GetDragDelta()
{
	return drag_delta;
}

bool InputManager::IsKeyDown(unsigned char key)
{
	return key_state[key];
}

bool InputManager::IsMouseDown(int mouse)
{
	return mouse_state[mouse] == 0;
}

#include "InputManager.h"

bool InputManager::key_state[256];
bool InputManager::mouse_state[3] = { 1, 1, 1 };
int InputManager::click_pos[2];
int InputManager::mouse_pos[2];
int InputManager::drag_delta[2];
Event<int, int> InputManager::OnMouseDrag;
Event<int, int> InputManager::OnMouseMove;
Event<int, int, int, int> InputManager::OnMouseClick;

void InputManager::KeyboardInput(unsigned char key, bool down, int x, int y)
{
	key_state[key] = down;
}

void InputManager::MouseInput(int mouse_event, int state, int x, int y)
{
	mouse_state[mouse_event] = state;
	if (IsMouseDown(0))
	{
		glutSetCursor(GLUT_CURSOR_NONE);
		click_pos[0] = x;
		click_pos[1] = y;
	}
	else
	{
		glutSetCursor(GLUT_CURSOR_LEFT_ARROW);
	}
	mouse_pos[0] = x;
	mouse_pos[1] = y;
	OnMouseClick.fire(mouse_event, state, x, y);
}

void InputManager::MouseMotion(int x, int y)
{
	if (IsMouseDown(0))
	{
		drag_delta[0] = (click_pos[0] - x);
		drag_delta[1] = (click_pos[1] - y);
		glutWarpPointer(click_pos[0], click_pos[1]);
		OnMouseDrag.fire(drag_delta[0], drag_delta[1]);
	}
	mouse_pos[0] = x;
	mouse_pos[1] = y;
	OnMouseMove.fire(x, y);
}

int* InputManager::GetDragDelta()
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

btVector3 InputManager::GetMousePos()
{
	return btVector3(mouse_pos[0],mouse_pos[1],0);
}

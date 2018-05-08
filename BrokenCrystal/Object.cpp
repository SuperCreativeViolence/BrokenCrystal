#include "Object.h"
#include <stdio.h>

void Object::Rotate(vec3 euler)
{
	key_pitch += radians(euler.x);
	key_yaw += radians(euler.y);
	key_roll += radians(euler.z);
}

void Object::Rotate()
{
	rotation = rotation * quat(vec3(key_pitch, key_yaw, key_roll));
	rotation = normalize(rotation);
	key_pitch = 0;
	key_yaw = 0;
	key_roll = 0;
}

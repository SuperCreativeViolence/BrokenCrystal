#ifndef OBJECT_H
#define OBJECT_H
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace glm;

class Object
{

public:
	vec3 position;
	quat rotation;
	vec3 scale;

	mat4 GetViewMatrix()
	{
		Rotate();
		return toMat4(rotation);
	}

	void Rotate(vec3 euler);



private:
	float key_pitch, key_yaw, key_roll;

	void Rotate();
};

#endif
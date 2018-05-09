#ifndef OBJECT_H
#define OBJECT_H
#include <memory>
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
	Object();

	typedef std::unique_ptr<Object> p;
	static p Create() { return p(new Object); }

	vec3 position;
	quat rotation;
	vec3 scale;

	mat4 GetViewMatrix()
	{
		return view_matrix;
	}

	void Rotate(vec3 euler);
	void Rotate(float x, float y, float z);
	void Translate(vec3 vector);
	void Translate(float x, float y, float z);
	void Scale(vec3 vector);
	void Scale(float x, float y, float z);

	void UpdateView();

private:
	float key_pitch, key_yaw, key_roll;
	mat4 view_matrix;
};

#endif
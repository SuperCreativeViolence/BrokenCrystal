#ifndef CAMERA_H
#define CAMERA_H
#include "Object.h"

class Camera : public Object
{
public:
	typedef std::unique_ptr<Camera> p;
	static p Create(){ return p(new Camera);}

	Camera();

	float pitch, yaw, roll;

	void Rotate(vec3 euler) override;
	void Rotate(float x, float y, float z) override;

};

#endif
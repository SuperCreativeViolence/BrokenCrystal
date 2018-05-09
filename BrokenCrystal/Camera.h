#ifndef CAMERA_H
#define CAMERA_H
#include "Object.h"

class Camera : public Object
{
public:
	typedef std::unique_ptr<Camera> p;
	Camera();
};

#endif
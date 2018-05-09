#ifndef CAMERA_H
#define CAMERA_H
#include "Object.h"

class Camera : public Object
{
public:
	typedef std::unique_ptr<Camera> p;
	static p Create(){ return p(new Camera);}

	Camera();
};

#endif
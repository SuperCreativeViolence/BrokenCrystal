#ifndef OPENGLMOTIONSTATE_H_
#define OPENGLMOTIONSTATE_H_

#include <btBulletCollisionCommon.h>

class OpenglMotionState : public btDefaultMotionState
{
public:
	OpenglMotionState(const btTransform &transform) : btDefaultMotionState(transform) { }

	void GetWorldTransform(btScalar* transform)
	{
		btTransform trans;
		getWorldTransform(trans);
		trans.getOpenGLMatrix(transform);
	}
};

#endif

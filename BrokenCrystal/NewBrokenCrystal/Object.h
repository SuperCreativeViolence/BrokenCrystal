#ifndef OBJECT_H
#define OBJECT_H

#include "Material.h"
#include "OpenglMotionState.h"
#include <btBulletDynamicsCommon.h>
#include <btBulletCollisionCommon.h>

#include <assert.h>

#define EPSILON 1e-4

struct ObjectIntersection
{
	bool hit;
	double u;
	btVector3 normal;
	Material material;
	ObjectIntersection(bool hit_ = false, double u_ = 0, const btVector3& normal_ = btVector3(0, 0, 0), Material material_ = Material());
};

class Object
{
public:
	Object(btCollisionShape* pShape, const btVector3 &position = btVector3(0,0,0), const btQuaternion &rotation = btQuaternion(0,0,1,1), Material material_ = Material(), float mass = 0);
	~Object();

	void GetTransform(btScalar* transform)
	{
		if (motionState)
			motionState->GetWorldTransform(transform);
	}

	btCollisionShape* GetShape()
	{
		return shape;
	}

	btRigidBody* GetRigidBody()
	{
		return body;
	}

	OpenglMotionState* GetMotionState()
	{
		return motionState;
	}

	btVector3 GetPosition()
	{
		assert(motionState);
		return motionState->GetWorldPosition();
	}

protected:
	virtual ObjectIntersection GetIntersection(const Ray& ray) = 0;

	btCollisionShape* shape;
	btRigidBody* body;
	OpenglMotionState* motionState;
	Material material;
};

class Box : public Object
{
public:


private:

};

class Sphere : public Object
{
public:
	Sphere(const btVector3 &position_, double radius_, float mass_, Material material_);

protected:
	virtual ObjectIntersection GetIntersection(const Ray& ray) override;

private:
	double radius;

};

#endif
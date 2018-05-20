#ifndef OBJECT_H
#define OBJECT_H

#include "Material.h"
#include "OpenglMotionState.h"
#include <BulletCollision\CollisionShapes\btShapeHull.h>
#include <btBulletCollisionCommon.h>
#include <LinearMath/btQuaternion.h>
#include "KDTree.h"

#include <iostream>
#include <assert.h>
#include <limits>

#define USE_KDTREE

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

	btQuaternion GetRotation()
	{
		assert(motionState);
		return motionState->GetWorldRotation();
	}

	Material GetMaterial()
	{
		return material;
	}

	virtual ObjectIntersection GetIntersection(const Ray& ray) = 0;

	btCollisionShape* shape;
	btRigidBody* body;
	OpenglMotionState* motionState;
	Material material;
};

class Sphere : public Object
{
public:
	Sphere(const btVector3 &position_, double radius_, float mass_, Material material_);
	virtual ObjectIntersection GetIntersection(const Ray& ray) override;

private:
	double radius;
};

class Mesh : public Object
{
public:
	Mesh(const btVector3& position_, std::vector<Triangle*> triangles_, float mass, Material material_);
	Mesh(const btVector3& position_, const char* filePath, float mass, Material material_);
	virtual ObjectIntersection GetIntersection(const Ray& ray) override;
	std::vector<Triangle*> GetTriangles() const;

private:
	std::vector<Triangle*> triangles;
	KDNode *node;
};

#endif
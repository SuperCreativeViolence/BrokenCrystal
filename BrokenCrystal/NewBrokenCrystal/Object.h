#ifndef OBJECT_H
#define OBJECT_H

#include "Material.h"
#include "OpenglMotionState.h"
#include <btBulletDynamicsCommon.h>
#include <BulletCollision\CollisionShapes\btShapeHull.h>
#include <btBulletCollisionCommon.h>
#include <LinearMath/btQuaternion.h>

#include <assert.h>
#include <limits>

#define EPSILON 1e-4

struct ObjectIntersection
{
	bool hit;
	double u;
	btVector3 normal;
	Material material;
	ObjectIntersection(bool hit_ = false, double u_ = 0, const btVector3& normal_ = btVector3(0, 0, 0), Material material_ = Material());
};


struct Triangle
{
public:
	Triangle(const btVector3 &pos1_, const btVector3 &pos2_, const btVector3 &pos3_, Material material_);
	ObjectIntersection GetIntersection(const Ray& ray, btTransform transform);
	Material GetMaterial();
	btVector3 pos[3];
	Material material;

private:
	double d;
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

class Box : public Object
{
public:
	Box(const btVector3 &position_, const btVector3 &halfExtents_, float mass_, Material material_);
	virtual ObjectIntersection GetIntersection(const Ray& ray) override;

private:
	btVector3 halfExtents;
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
	virtual ObjectIntersection GetIntersection(const Ray& ray) override;
	std::vector<Triangle*> GetTriangles() const;

private:
	std::vector<Triangle*> triangles;
};

#endif
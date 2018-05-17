#include "Object.h"

ObjectIntersection::ObjectIntersection(bool hit_, double u_, const btVector3& normal_, Material material_)
{
	hit = hit_;
	u = u_;
	normal = normal_;
	material = material_;
}

Object::Object(btCollisionShape* pShape, const btVector3 &position, const btQuaternion &rotation, Material material_, float mass)
{
	shape = pShape;

	btTransform transform = btTransform::getIdentity();
	transform.setOrigin(position);
	transform.setRotation(rotation);
	motionState = new OpenglMotionState(transform);

	btVector3 localInteria(0, 0, 0);
	if (mass != 0.0f)
		pShape->calculateLocalInertia(mass, localInteria);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, motionState, pShape, localInteria);
	body = new btRigidBody(cInfo);
	material = material_;
}

Object::~Object()
{
	delete body;
	delete motionState;
	delete shape;
}

Box::Box(const btVector3 &position_, const btVector3 &halfExtents_, float mass_, Material material_) : Object(new btBoxShape(halfExtents_), position_, btQuaternion(0, 0, 1, 1), material_, mass_)
{
	halfExtents = halfExtents_;
}

ObjectIntersection Box::GetIntersection(const Ray& ray)
{
	bool hit = false;
	double u = 0;
	btVector3 normal = btVector3(0, 0, 0);
	double tmin, tmax, tymin, tymax, tzmin, tzmax;
	int tminIndex = 0;
	int tmaxIndex = 0;
	btVector3 position = GetPosition();
	btQuaternion rotation = GetRotation();
	btVector3 min = position - halfExtents;
	btVector3 max = position + halfExtents;
	
	tmin = (min[0] - ray.origin[0]) / ray.direction[0];
	tmax = (max[0] - ray.origin[0]) / ray.direction[0];

	if (tmin > tmax)
		btSwap(tmin, tmax);

	tymin = (min[1] - ray.origin[1]) / ray.direction[1];
	tymax = (max[1] - ray.origin[1]) / ray.direction[1];

	if (tymin > tymax)
		btSwap(tymin, tymax);

	if ((tmin > tymax) || (tymin > tmax))
		return ObjectIntersection(hit, u, normal, material);

	if (tymin > tmin)
	{
		tminIndex = 1;
		tmin = tymin;
	}

	if (tymax < tmax)
	{
		tmaxIndex = 1;
		tmax = tymax;
	}

	tzmin = (min[2] - ray.origin[2]) / ray.direction[2];
	tzmax = (max[2] - ray.origin[2]) / ray.direction[2];

	if (tzmin > tzmax)
		btSwap(tzmin, tzmax);

	if ((tmin > tzmax) || (tzmin > tmax))
		return ObjectIntersection(hit, u, normal, material);

	if (tzmin > tmin)
	{
		tminIndex = 2;
		tmin = tzmin;
	}

	if (tzmax < tmax)
	{
		tmaxIndex = 2;
		tmax = tzmax;
	}

	hit = true;
	static btVector3 normals[3] = { btVector3(1,0,0), btVector3(0,1,0), btVector3(0,0,1) };

	if (tmin <= 0)
	{
		u = tmax;
		normal = normals[tmaxIndex];
	}
	else
	{
		u = tmin;
		normal = normals[tminIndex];
	}

	btTransform transform = btTransform::getIdentity();
	transform.setRotation(rotation);
	transform.setOrigin(position);

	normal = transform * normal;

	if (normal.dot(ray.direction) > 0)
		normal = normal * -1;

	normal = normal.normalize();

	return ObjectIntersection(hit, u, normal, material);
}

Sphere::Sphere(const btVector3 &position_, double radius_, float mass_, Material material_) : Object(new btSphereShape(radius_), position_, btQuaternion(0,0,1,1), material_, mass_)
{
	radius = radius_;
}

ObjectIntersection Sphere::GetIntersection(const Ray& ray)
{
	bool hit = false;
	double distance = 0;
	btVector3 normal = btVector3(0, 0, 0);
	btVector3 position = GetPosition();

	btVector3 op = position - ray.origin;
	double t;
	double b = op.dot(ray.direction);
	double det = b * b - op.dot(op) + radius * radius;

	if (det < 0) return ObjectIntersection(hit, distance, normal, material);
	else det = sqrt(det);

	distance = (t = b - det) > EPSILON ? t : ((t = b + det) > EPSILON ? t : 0);
	if (distance != 0)
	{
		hit = true;
		normal = ((ray.origin + ray.direction * distance) - position).normalize();
	}
	return ObjectIntersection(hit, distance, normal, material);
}
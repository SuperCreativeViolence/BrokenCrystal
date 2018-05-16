#include "Object.h"

ObjectIntersection::ObjectIntersection(bool hit_, double u_, const btVector3& normal_, Material material_)
{
	hit = hit_;
	u = u_;
	normal = normal_;
	material = material_;
}

Object::Object(btCollisionShape* pShape, const btVector3 &position /*= btVector3(0,0,0)*/, const btQuaternion &rotation /*= btQuaternion(0,0,1,1)*/, Material material_ /*= Material()*/, float mass /*= 0*/)
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

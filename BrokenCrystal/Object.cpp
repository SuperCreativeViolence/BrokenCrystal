#include "Object.h"

ObjectIntersection::ObjectIntersection(bool hit_, double u_, const btVector3& n_, Material m_)
{
	hit = hit_, u = u_, n = n_, m = m_;
}


Object::Object()
{

}

Object::Object(btCollisionShape* pShape, float mass, const btVector3 &color, const btVector3 &initialPosition, const btQuaternion &initialRotation)
{
	bt_Shape = pShape;
	bt_Color = color;

	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(initialPosition);
	transform.setRotation(initialRotation);

	bt_MotionState = new OpenGLMotionState(transform);

	btVector3 localInertia(0, 0, 0);

	if (mass != 0.0f)
		pShape->calculateLocalInertia(mass, localInertia);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, bt_MotionState, pShape, localInertia);

	bt_Body = new btRigidBody(cInfo);
}

Object::~Object()
{
	delete bt_Body;
	delete bt_MotionState;
	delete bt_Shape;
}

btQuaternion Object::GetWorldRotation()
{
	return bt_MotionState->GetWorldRotation();
}

btVector3 Object::GetWorldPosition()
{
	return bt_MotionState->GetWorldPosition();
}

btVector3 Object::GetWorldEulerRotation()
{
	btQuaternion q = GetWorldRotation();
	btScalar yaw, pitch, roll;
	q.getEulerZYX(yaw, pitch, roll);
	return btVector3(pitch, yaw, roll);
}

void Object::SetRotation(btQuaternion quat)
{
	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(btVector3(0,0,0));
	btVector3 offset = GetWorldPosition();
	transform.setRotation(quat);
	transform.setOrigin(offset);
	bt_MotionState->setWorldTransform(transform);
}

void Object::Rotate(vec3 euler)
{
	Rotate(euler.x, euler.y ,euler.z);
}

void Object::Rotate(float x, float y, float z)
{

}

void Object::Translate(const btVector3& vector, bool isLocal)
{
	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(GetWorldPosition() + vector);
	transform.setRotation(GetWorldRotation());
	bt_MotionState->setWorldTransform(transform);
}

void Object::Translate(float x, float y, float z, bool isLocal)
{
	Translate(btVector3(x, y, z), isLocal);
}

void Object::Scale(vec3 vector)
{

}

void Object::Scale(float x, float y, float z)
{
	Scale(vec3(x, y, z));
}

void Object::LookAt(vec3 pos)
{

}

void Object::LookAt(float x, float y, float z)
{
	LookAt(vec3(x, y, z));
}

btCollisionShape* Object::GetShape()
{
	return bt_Shape;
}

btRigidBody* Object::GetRigidBody()
{
	return bt_Body;
}

btMotionState* Object::GetMotionState()
{
	return bt_MotionState;
}

void Object::GetTransform(btScalar* transform)
{
	if (bt_MotionState)
		bt_MotionState->GetWorldTransform(transform);
}

btVector3 Object::GetColor()
{
	return bt_Color;
}

void Object::SetColor(const btVector3 &color)
{
	bt_Color = color;
}

Material Object::GetMaterial()
{
	return material;
}

Sphere::Sphere(const btVector3 & position_, double radius_, double mass_, Material material_) : Object(new btSphereShape(radius_), mass_, btVector3(0, 0, 0), position_)
{
	radius = radius_;
	material = material_;
}

double Sphere::get_radius()
{
	return radius;
}

ObjectIntersection Sphere::get_intersection(const Ray & ray)
{
	bool hit = false;
	double distance = 0;
	btVector3 n = btVector3(0, 0, 0);
	btVector3 position = GetWorldPosition();
	btVector3 op = position - ray.origin;
	double t, eps = 1e-4, b = op.dot(ray.direction), det = b * b - op.dot(op) + radius * radius;
	if (det < 0) return ObjectIntersection(hit, distance, n, material);
	else det = sqrt(det);
	distance = (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
	if (distance != 0)
	{
		hit = true;
		n = ((ray.origin + ray.direction * distance) - position).normalize();
	}

	return ObjectIntersection(hit, distance, n, material);
}

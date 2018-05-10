#include "Object.h"

Object::Object()
{
	position = vec3();
	rotation = quat();
	scale = vec3(1);
	isdirty_update = true;
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
	isdirty_update = true;
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

void Object::SetRotation(const quat& rot)
{
	rotation = rot;
}

quat Object::GetRotation() const
{
	return rotation;
}

void Object::Rotate(vec3 euler)
{
	Rotate(euler.x, euler.y ,euler.z);
}

void Object::Rotate(float x, float y, float z)
{
	key_pitch += radians(x);
	key_yaw += radians(y);
	key_roll += radians(z);

	isdirty_update = true;
}

void Object::Translate(vec3 vector, bool isLocal)
{
	position += isLocal ? rotation * vector : vector;
	isdirty_update = true;
}

void Object::Translate(float x, float y, float z, bool isLocal)
{
	Translate(vec3(x, y, z), isLocal);
}

void Object::Scale(vec3 vector)
{
	scale += vector;
	isdirty_update = true;
}

void Object::Scale(float x, float y, float z)
{
	Scale(vec3(x, y, z));
}

void Object::LookAt(vec3 pos)
{
	SetRotation(lookAt(position, -pos, vec3(0, 1, 0)));
	isdirty_update = true;
}

void Object::LookAt(float x, float y, float z)
{
	LookAt(vec3(x, y, z));
}

void Object::UpdateView()
{
	if (!isdirty_update)
		return;

	mat4 translation_mat = translate(-position);

	rotation = rotation * quat(vec3(key_pitch, key_yaw, key_roll));
	rotation = normalize(rotation);
	key_pitch = key_yaw = key_roll = 0;
	mat4 rotation_mat = transpose(toMat4(rotation));

	mat4 scale_mat;
	scale_mat = glm::scale(scale_mat, scale);

	view_matrix = translation_mat * rotation_mat * scale_mat;
	isdirty_update = false;
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

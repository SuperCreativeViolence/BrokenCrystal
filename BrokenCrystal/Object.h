#ifndef OBJECT_H
#define OBJECT_H
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <BulletPhysics/btBulletDynamicsCommon.h>
#include "OpenGLMotionState.h"


using namespace glm;

class Object
{

public:
	Object();
	Object(btCollisionShape* pShape, float mass, const btVector3 &color,
		const btVector3 &initialPosition = btVector3(0, 0, 0),
		const btQuaternion &initialRotation = btQuaternion(0, 0, 1, 1));
	~Object();

	typedef std::unique_ptr<Object> p;
	static p Create() { return p(new Object); }

	vec3 position;
	vec3 scale;

	mat4 GetViewMatrix()
	{
		UpdateView();
		return view_matrix;
	}

	btQuaternion GetWorldRotation();
	btVector3 GetWorldPosition();
	btVector3 GetWorldEulerRotation();
	void SetRotation(const glm::quat& rot);
	quat GetRotation() const;

	virtual void Rotate(vec3 euler);
	virtual void Rotate(float x, float y, float z);
	void Translate(vec3 vector, bool isLocal = true);
	void Translate(float x, float y, float z, bool isLocal = true);
	void Scale(vec3 vector);
	void Scale(float x, float y, float z);

	virtual void LookAt(vec3 position);
	virtual void LookAt(float x, float y, float z);

	virtual void UpdateView();

	btCollisionShape* GetShape();
	btRigidBody* GetRigidBody();
	btMotionState* GetMotionState();
	void GetTransform(btScalar* transform);
	btVector3 GetColor();
	void SetColor(const btVector3 &color);

protected:
	quat rotation;
	mat4 view_matrix;
	bool isdirty_update;
	float key_pitch, key_yaw, key_roll;

	btCollisionShape* bt_Shape;
	btRigidBody* bt_Body;
	OpenGLMotionState* bt_MotionState;
	btVector3 bt_Color;

};

#endif
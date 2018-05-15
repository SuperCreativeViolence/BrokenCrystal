#ifndef OBJECT_H
#define OBJECT_H
#include <memory>
#include "Material.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <gl/freeglut.h>
#include <BulletPhysics/btBulletDynamicsCommon.h>

#include "OpenGLMotionState.h"

// 나중에 지워야함
using namespace glm;

struct ObjectIntersection
{
	bool hit;	// If there was an intersection
	double u;	// Distance to intersection along ray
	btVector3 n;		// Normal of intersected face
	Material m;	// Material of intersected face

	ObjectIntersection(bool hit_ = false, double u_ = 0, const btVector3& n_ = btVector3(0, 0, 0), Material m_ = Material());
};

class Object
{

public:
	Object();
	Object(btCollisionShape* pShape, float mass,
		   const btVector3 &color = btVector3(0, 0, 0),
		   const btVector3 &initialPosition = btVector3(0, 0, 0),
		   const btQuaternion &initialRotation = btQuaternion(0, 0, 1, 1));
	~Object();

	vec3 scale;

	btScalar* GetViewMatrix()
	{
		btScalar transform[16];
		bt_MotionState->GetWorldTransform(transform);
		return transform;
	}

	btQuaternion GetWorldRotation();
	btVector3 GetWorldPosition();
	btVector3 GetWorldEulerRotation();
	void SetRotation(btQuaternion quat);

	virtual void Rotate(vec3 euler);
	virtual void Rotate(float x, float y, float z);
	void Translate(const btVector3& vector, bool isLocal = true);
	void Translate(float x, float y, float z, bool isLocal = true);
	void Scale(vec3 vector);
	void Scale(float x, float y, float z);

	virtual void LookAt(vec3 position);
	virtual void LookAt(float x, float y, float z);

	btCollisionShape* GetShape();
	btRigidBody* GetRigidBody();
	btMotionState* GetMotionState();
	void GetTransform(btScalar* transform);
	btVector3 GetColor();
	void SetColor(const btVector3 &color);

	// 레이트레이싱과 관련
	virtual bool Intersect(const btVector3 &from, const btVector3 &to, float &t)
	{
		return true;
	};

	// 패스트레이싱과 관련
	virtual ObjectIntersection get_intersection(const Ray &r) = 0;
	Material GetMaterial();


protected:
	btCollisionShape* bt_Shape;
	btRigidBody* bt_Body;
	OpenGLMotionState* bt_MotionState;
	btVector3 bt_Color;
	Material material;

};

class Sphere : public Object
{
public:
	Sphere(const btVector3 & position_, double radius_, double mass_, Material material_);
	double get_radius();
	virtual ObjectIntersection get_intersection(const Ray &ray) override;

private:
	double radius;
};

#endif
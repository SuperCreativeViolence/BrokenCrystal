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

Triangle::Triangle(const btVector3 &pos1_, const btVector3 &pos2_, const btVector3 &pos3_, Material material_)
{
	pos[0] = pos1_;
	pos[1] = pos2_;
	pos[2] = pos3_;
	material = material_;
}

ObjectIntersection Triangle::GetIntersection(const Ray& ray, btTransform transform)
{
	bool hit = false;
	float u, v, t = 0;

	btVector3 pos0 = transform * pos[0];
	btVector3 pos1 = transform * pos[1];
	btVector3 pos2 = transform * pos[2];
	btVector3 normal = (pos1 - pos0).cross(pos2 - pos0).normalize();

	btVector3 v0v1 = pos1 - pos0;
	btVector3 v0v2 = pos2 - pos0;
	btVector3 pvec = ray.direction.cross(v0v2);
	float det = v0v1.dot(pvec);
	if (det < EPSILON) return ObjectIntersection(hit, t, normal, material);

	btVector3 tvec = ray.origin - pos0;
	u = tvec.dot(pvec);
	if (u < 0 || u > det) return ObjectIntersection(hit, t, normal, material);

	btVector3 qvec = tvec.cross(v0v1);
	v = ray.direction.dot(qvec);
	if (v < 0 || u + v > det) return ObjectIntersection(hit, t, normal, material);

	t = v0v2.dot(qvec) / det;

	if (t < EPSILON) return ObjectIntersection(hit, t, normal, material);

	hit = true;
	return ObjectIntersection(hit, t, normal, material);
}

Material Triangle::GetMaterial()
{
	return material;
}

Mesh::Mesh(const btVector3 & position_, std::vector<Triangle*> triangles_, float mass, Material material_) : Object(new btEmptyShape(), position_, btQuaternion(0,0,1,1), material_, mass)
{
	triangles = triangles_;
	btTriangleMesh* triangleMesh = new btTriangleMesh();
	for (auto & triangle : triangles)
	{
		triangleMesh->addTriangle(triangle->pos[0], triangle->pos[1], triangle->pos[2]);
	}

	btConvexShape* tempShape = new btConvexTriangleMeshShape(triangleMesh);
	btShapeHull* hull = new btShapeHull(tempShape);
	btScalar margin = tempShape->getMargin();
	hull->buildHull(margin);
	tempShape->setUserPointer(hull);
	shape = tempShape;
	btVector3 localInteria(0, 0, 0);
	if (mass != 0.0f)
		shape->calculateLocalInertia(mass, localInteria);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, motionState, shape, localInteria);
	body = new btRigidBody(cInfo);
}

ObjectIntersection Mesh::GetIntersection(const Ray& ray)
{
	float tNear = std::numeric_limits<float>::max();
	btTransform transform = body->getWorldTransform();
	ObjectIntersection intersection = ObjectIntersection();
	for (auto & triangle : triangles)
	{
		float u, v;
		ObjectIntersection temp = triangle->GetIntersection(ray, transform);
		if (temp.hit && temp.u < tNear)
		{
			tNear = temp.u;
			intersection = temp;
		}
	}
	return intersection;
}

std::vector<Triangle*> Mesh::GetTriangles() const
{
	return triangles;
}

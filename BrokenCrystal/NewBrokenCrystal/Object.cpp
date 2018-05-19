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
	//bool hit = false;
	//double u = 0;
	//btVector3 normal = btVector3(0, 0, 0);
	//double tmin, tmax, tymin, tymax, tzmin, tzmax;
	//int tminIndex = 0;
	//int tmaxIndex = 0;

	//btVector3 position = GetPosition();
	//btQuaternion rotation = GetRotation();
	//btTransform transform = btTransform::getIdentity();
	//transform.setRotation(rotation);
	//transform.setOrigin(position);

	//btVector3 min = position - halfExtents;
	//btVector3 max = position + halfExtents;

	//tmin = (min[0] - ray.origin[0]) / ray.direction[0];
	//tmax = (max[0] - ray.origin[0]) / ray.direction[0];

	//if (tmin > tmax)
	//	btSwap(tmin, tmax);

	//tymin = (min[1] - ray.origin[1]) / ray.direction[1];
	//tymax = (max[1] - ray.origin[1]) / ray.direction[1];

	//if (tymin > tymax)
	//	btSwap(tymin, tymax);

	//if ((tmin > tymax) || (tymin > tmax))
	//	return ObjectIntersection(hit, u, normal, material);

	//if (tymin > tmin)
	//{
	//	tminIndex = 1;
	//	tmin = tymin;
	//}

	//if (tymax < tmax)
	//{
	//	tmaxIndex = 1;
	//	tmax = tymax;
	//}

	//tzmin = (min[2] - ray.origin[2]) / ray.direction[2];
	//tzmax = (max[2] - ray.origin[2]) / ray.direction[2];

	//if (tzmin > tzmax)
	//	btSwap(tzmin, tzmax);

	//if ((tmin > tzmax) || (tzmin > tmax))
	//	return ObjectIntersection(hit, u, normal, material);

	//if (tzmin > tmin)
	//{
	//	tminIndex = 2;
	//	tmin = tzmin;
	//}

	//if (tzmax < tmax)
	//{
	//	tmaxIndex = 2;
	//	tmax = tzmax;
	//}

	//hit = true;
	//static btVector3 normals[3] = { btVector3(1,0,0), btVector3(0,1,0), btVector3(0,0,1) };

	//if (tmin <= 0)
	//{
	//	u = tmax;
	//	normal = normals[tmaxIndex];
	//}
	//else
	//{
	//	u = tmin;
	//	normal = normals[tminIndex];
	//}

	//normal = transform * normal;

	//if (normal.dot(ray.direction) > 0)
	//	normal = normal * -1;

	//normal = normal.normalize();

	//return ObjectIntersection(hit, u, normal, material);

	//bool hit = false;
	//double u = 0;
	//btVector3 normal = btVector3(0, 0, 0);

	//btVector3 position = GetPosition();
	//btQuaternion rotation = GetRotation();

	//btVector3 S = ray.origin;
	//btVector3 c = ray.direction;

	//double D = halfExtents[0];

	//double t1, t2;
	//double tnear, tfar;
	//char tnear_index = 0, tfar_index = 0;

	//// intersect with the X planes
	//if (c.x() == 0 && (S.x() < -D || S.x() > D))
	//	return ObjectIntersection(hit, u, normal, material);
	//else
	//{
	//	printf("X ");
	//	tnear = (-D - S.x()) / c.x();
	//	tfar = (D - S.x()) / c.x();

	//	if (tnear > tfar)
	//		btSwap(tnear, tfar);

	//	if (tnear > tfar)
	//		return ObjectIntersection(hit, u, normal, material);
	//	if (tfar < 0)
	//		return ObjectIntersection(hit, u, normal, material);
	//}

	//D = halfExtents[1];
	//// intersect with the Y planes
	//if (c.y() == 0 && (S.y() < -D || S.y() > D))
	//	return ObjectIntersection(hit, u, normal, material);
	//else
	//{
	//	printf("Y ");
	//	t1 = (-D - S.y()) / c.y();
	//	t2 = (D - S.y()) / c.y();

	//	if (t1 > t2)
	//		btSwap(t1, t2);

	//	if (t1 > tnear)
	//	{
	//		tnear = t1;
	//		tnear_index = 1;
	//	}

	//	if (t2 < tfar)
	//	{
	//		tfar = t2;
	//		tfar_index = 1;
	//	}

	//	if (tnear > tfar)
	//		return ObjectIntersection(hit, u, normal, material);
	//	if (tfar < 0)
	//		return ObjectIntersection(hit, u, normal, material);
	//}

	//D = halfExtents[2];
	//// intersect with the Z planes
	//if (c.z() == 0 && (S.z() < -D || S.z() > D))
	//	return ObjectIntersection(hit, u, normal, material);
	//else
	//{
	//	printf("Z ");
	//	t1 = (-D - S.z()) / c.z();
	//	t2 = (D - S.z()) / c.z();

	//	if (t1 > t2)
	//		btSwap(t1, t2);

	//	if (t1 > tnear)
	//	{
	//		tnear = t1;
	//		tnear_index = 2;
	//	}

	//	if (t2 < tfar)
	//	{
	//		tfar = t2;
	//		tfar_index = 2;
	//	}

	//	if (tnear > tfar)
	//		return ObjectIntersection(hit, u, normal, material);
	//	if (tfar < 0)
	//		return ObjectIntersection(hit, u, normal, material);
	//}

	//const btVector3 normals[3] = { btVector3(1,0,0), btVector3(0,1,0), btVector3(0,0,1) };

	//if (tnear <= 0)
	//{
	//	u = tfar;
	//	normal = normals[tfar_index];
	//}
	//else
	//{
	//	u = tnear;
	//	normal = normals[tnear_index];
	//}

	//btTransform transform = btTransform::getIdentity();
	//transform.setRotation(rotation);
	//transform.setOrigin(position);

	//normal = transform * normal;

	//if (normal.dot(c) > 0)
	//	normal = -1 * normal;

	//normal.normalize();
	//hit = true;
	//return ObjectIntersection(hit, u, normal, material);

	bool hit = false;
	double u = 0;
	btVector3 normal = btVector3(0, 0, 0);

	btVector3 position = GetPosition();
	btQuaternion rotation = GetRotation();
	btTransform transform = btTransform::getIdentity();
	transform.setRotation(rotation);
	transform.setOrigin(position);

	btScalar tmin, tmax, tymin, tymax, tzmin, tzmax;
	btVector3 bounds[2] = { (position - halfExtents), (position + halfExtents) };

	int tminIndex = 0;
	int tmaxIndex = 0;

	tmin = (bounds[ray.sign[0]].x() - ray.origin.x()) * ray.direction_inv.x();
	tmax = (bounds[1 - ray.sign[0]].x() - ray.origin.x()) * ray.direction_inv.x();
	tymin = (bounds[ray.sign[1]].y() - ray.origin.y()) * ray.direction_inv.y();
	tymax = (bounds[1 - ray.sign[1]].y() - ray.origin.y()) * ray.direction_inv.y();

	if ((tmin > tymax) || (tymin > tmax))
		return ObjectIntersection(hit, u, normal, material);
	if (tymin > tmin)
		tmin = tymin, tminIndex = 1;
	if (tymax < tmax)
		tmax = tymax, tmaxIndex = 1;

	tzmin = (bounds[ray.sign[2]].z() - ray.origin.z()) * ray.direction_inv.z();
	tzmax = (bounds[1 - ray.sign[2]].z() - ray.origin.z()) * ray.direction_inv.z();

	if ((tmin > tzmax) || (tzmin > tmax))
		return ObjectIntersection(hit, u, normal, material);
	if (tzmin > tmin)
		tmin = tzmin, tminIndex = 2;
	if (tzmax < tmax)
		tmax = tzmax, tmaxIndex = 2;

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

	//normal = transform * normal;
	//normal = normal.normalized();

	if (normal.dot(ray.direction) > 0)
		normal = normal * -1;

	hit = true;
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

	if (t < 0) return ObjectIntersection(hit, t, normal, material);

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

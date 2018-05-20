#include "Triangle.h"


Triangle::Triangle(const btVector3 &pos1_, const btVector3 &pos2_, const btVector3 &pos3_, Material material_)
{
	pos[0] = pos1_;
	pos[1] = pos2_;
	pos[2] = pos3_;
	material = material_;
}

Triangle::Triangle(const btVector3 &pos1_, const btVector3 &pos2_, const btVector3 &pos3_, const btVector3 &t0_, const btVector3 &t1_, const btVector3 &t2_, Material material_)
	: Triangle(pos1_, pos2_, pos3_, material_)
{
	tex[0] = t0_;
	tex[1] = t1_;
	tex[2] = t2_;
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

AABBox Triangle::GetBoundingBox()
{
	btVector3 min = btVector3(
		std::min(std::min(pos[0].x(), pos[1].x()), pos[2].x()),
		std::min(std::min(pos[0].y(), pos[1].y()), pos[2].y()),
		std::min(std::min(pos[0].z(), pos[1].z()), pos[2].z())
	);
	btVector3 max = btVector3(
		std::max(std::max(pos[0].x(), pos[1].x()), pos[2].x()),
		std::max(std::max(pos[0].y(), pos[1].y()), pos[2].y()),
		std::max(std::max(pos[0].z(), pos[1].z()), pos[2].z())
	);

	return AABBox(min, max);
}

bool Triangle::Intersect(Ray ray, double &t, double tmin, btVector3 &normal, btTransform transform) const
{
	float u, v, dis = 0;

	btVector3 pos0 = transform * pos[0];	
	btVector3 pos1 = transform * pos[1];
	btVector3 pos2 = transform * pos[2];
	btVector3 norm = (pos1 - pos0).cross(pos2 - pos0).normalize();

	btVector3 v0v1 = pos1 - pos0;
	btVector3 v0v2 = pos2 - pos0;
	btVector3 pvec = ray.direction.cross(v0v2);
	float det = v0v1.dot(pvec);
	if (det < EPSILON) return false;

	btVector3 tvec = ray.origin - pos0;
	u = tvec.dot(pvec);
	if (u < 0 || u > det) return false;

	btVector3 qvec = tvec.cross(v0v1);
	v = ray.direction.dot(qvec);
	if (v < 0 || u + v > det) return false;

	dis = v0v2.dot(qvec) / det;

	if (dis < EPSILON) return false;

	t = dis;
	normal = norm;

	return true;
}

btVector3 Triangle::GetMidPoint()
{
	return (pos[0] + pos[1] + pos[2]) / 3;
}

btVector3 Triangle::GetBarycentric(btVector3 position)
{
	btVector3 e1 = pos[1] - pos[0];
	btVector3 e2 = pos[2] - pos[0];

	btVector3 v2_ = position - pos[0];
	double d00 = e1.dot(e1);
	double d01 = e1.dot(e2);
	double d11 = e2.dot(e2);
	double d20 = v2_.dot(e1);
	double d21 = v2_.dot(e2);
	double d = d00 * d11 - d01 * d01;
	double v = (d11*d20 - d01 * d21) / d;
	double w = (d00*d21 - d01 * d20) / d;
	double u = 1 - v - w;
	return btVector3(u, v, w);
}

btVector3 Triangle::GetColorAt(btVector3 pos)
{
	//btVector3 b = GetBarycentric(pos);
	//btVector3 c = btVector3();
	//c = c + (t0 * b[0]);
	//c = c + (t1 * b[1]);
	//c = c + (t2 * b[z]);

	//return m->get_colour_at(c.[0], c.[1]);
	if (material.GetType() == EMIT)
		return material.GetEmission();
	return material.GetColor();
}

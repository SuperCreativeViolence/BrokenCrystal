#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Ray.h"
#include "Intersection.h"
#include "AABBox.h"

struct Triangle
{
public:
	Triangle(const btVector3 &pos1_, const btVector3 &pos2_, const btVector3 &pos3_, Material material_);
	Triangle(const btVector3 &pos1_, const btVector3 &pos2_, const btVector3 &pos3_, const btVector3 &t0_, const btVector3 &t1_, const btVector3 &t2_, Material material_);
	ObjectIntersection GetIntersection(const Ray& ray, btTransform transform);
	Material GetMaterial();

	// KDTree
	AABBox get_bounding_box();
	bool Intersect(Ray ray, double &t, double tmin, btVector3 &norma, btTransform transform) const;

	btVector3 get_midpoint();
	btVector3 GetBarycentric(btVector3 pos);
	btVector3 GetColorAt(btVector3 pos);


	btVector3 pos[3];
	btVector3 tex[3];
private:
	double d;
	Material material;

};

#endif
#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "Material.h"
#include <btBulletDynamicsCommon.h>


struct ObjectIntersection
{
	bool hit;
	double u;
	btVector3 normal;
	Material material;
	ObjectIntersection(bool hit_ = false, double u_ = 0, const btVector3& normal_ = btVector3(0, 0, 0), Material material_ = Material());
};

#endif

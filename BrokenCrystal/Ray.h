#include <BulletPhysics\LinearMath\btVector3.h>
#ifndef RAY_H
#define RAY_H

struct Ray
{
	btVector3 origin, direction, direction_inv;
	Ray(const btVector3 & o_, const btVector3 & d_) : origin(o_), direction(d_)
	{
		direction_inv = btVector3(
			1. / direction[0],
			1. / direction[1],
			1. / direction[2]
		);
	}
};

#endif
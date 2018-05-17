#include <LinearMath\btVector3.h>
#ifndef RAY_H
#define RAY_H

struct Ray
{
	btVector3 origin;
	btVector3 direction;
	btVector3 direction_inv;

	Ray(const btVector3& origin_, const btVector3& direction_) : origin(origin_), direction(direction_)
	{
		direction_inv = btVector3
		(
			1.0 / direction.getX(),
			1.0 / direction.getY(),
			1.0 / direction.getZ()
		);
 	}
};

#endif
#ifndef RAY_H
#define RAY_H

#include <LinearMath\btVector3.h>

struct Ray
{
	btVector3 origin;
	btVector3 direction;
	btVector3 direction_inv;
	int sign[3];

	Ray(const btVector3& origin_, const btVector3& direction_) : origin(origin_), direction(direction_)
	{
		direction_inv = btVector3
		(
			1.0 / direction.getX(),
			1.0 / direction.getY(),
			1.0 / direction.getZ()
		);
		sign[0] = (direction_inv[0] < 0);
		sign[1] = (direction_inv[1] < 0);
		sign[2] = (direction_inv[2] < 0);
 	}
};

#endif
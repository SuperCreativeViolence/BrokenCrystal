#ifndef CAMERACU_H
#define CAMERACU_H

#include "cutil_math.h"
#include "curand.h"
#include "curand_kernel.h"

#define CU_SIMD_PI float(3.1415926535897932384626433832795029)
#define CU_SIMD_2_PI (float(2.0) * CU_SIMD_PI )
#define CU_SIMD_RADS_PER_DEG (CU_SIMD_2_PI / float(360.0))

struct RayCU
{
	float3 origin;
	float3 direction;
	float3 direction_inv;

	__host__ __device__ RayCU(float3 origin_, float3 direction_) : origin(origin_), direction(direction_)
	{
		direction_inv = make_float3
		(
			1.0 / direction.x,
			1.0 / direction.y,
			1.0 / direction.z
		);
	}

};

class CameraCU
{
public:
	CameraCU(){}
	~CameraCU(){}

	__device__ RayCU GetRay(curandState* randState, int x, int y, int sx, int sy, int dof)
	{
		const double r1 = 2.0 * curand_uniform(randState);
		const double r2 = 2.0 * curand_uniform(randState);

		double dx;
		if (r1 < 1.0)
			dx = sqrt(r1) - 1.0;
		else
			dx = 1.0 - sqrt(2.0 - r1);

		double dy;
		if (r2 < 1.0)
			dy = sqrt(r2) - 1.0;
		else
			dy = 1.0 - sqrt(2.0 - r2);

		float3 wDir = normalize(-direction);
		float3 uDir = normalize(cross(upVector, wDir));
		float3 vDir = cross(wDir, -uDir);

		float top = tan(fov * 0.5 * CU_SIMD_RADS_PER_DEG);
		float right = aspectRatio * top;
		float bottom = -top;
		float left = -right;

		float imPlaneUPos = left + (right - left)*(((float)x + sx + dx + 0.5f) / (float)width);
		float imPlaneVPos = bottom + (top - bottom)*(((float)y + sy + dy + 0.5f) / (float)height);

		RayCU result = RayCU(position, (normalize(imPlaneUPos*uDir + imPlaneVPos * vDir - wDir)));
		/*
		if (dof)
		{
			double u1 = (erand48() * 2.0) - 1.0;
			double u2 = (erand48() * 2.0) - 1.0;

			double fac = (double)(2 * 3.14159265358979323846 * u2);

			btVector3 offset = aperture * btVector3(u1 * cos(fac), u1 * sin(fac), 0.0);
			btVector3 focalPlaneIntersection = result.origin + result.direction * (focalLength / direction.dot(result.direction));
			result.origin = result.origin + offset;
			result.direction = (focalPlaneIntersection - result.origin).normalize();
		}*/

		return result;

	}

	float3 position;
	float3 target;
	float3 direction;
	float nearPlane;
	float farPlane;
	float3 upVector;
	float distance;
	float pitch;
	float yaw;
	float fov;
	float aspectRatio;

	int width;
	int height;

	// dof
	int aperture;
	int focalLength;

private:

};
#endif
#ifndef TRACEPATH_H
#define TRACEPATH_H



#include "curand.h"
#include "curand_kernel.h"
#include "cutil_math.h"

#include "CameraCU.h"
#include "MaterialType.h"
#include <iostream>


#define FLT_MAX_CU	3.402823466e+38F 

#define EPSILON_CU 1e-10

#define TRACE_SAMPLES 4

#define THREAD_NUM 



struct ObjectIntersectionCU
{
	bool hit;
	double u;
	float3 normal;
	MaterialType material;
	float3 color;
	float3 emission;
	__host__ __device__ ObjectIntersectionCU(bool hit_ = false, double u_ = 0, const float3& normal_ = make_float3(0.0f, 0.0f, 0.0f), MaterialType material_ = DIFF, const float3& color_ = make_float3(0.0f, 0.0f, 0.0f), const float3& emission_ = make_float3(0.0f, 0.0f, 0.0f))
	{
		hit = hit_;
		u = u_;
		normal = normal_;
		material = material_;
		color = color_;
		emission = emission_;
	}
};

struct ObjectCU
{
	unsigned int triangles_size;
	float3* triangles_p;
	int material;
	float3 color;
	float3 emission;
};


class TracePath
{
public:
	TracePath() {}
	~TracePath(){}
	float3* RenderPathCU(ObjectCU* object_list, int num_objects, CameraCU* camera, int width, int height);
};


#endif // !TRACEPATH_CUH*/
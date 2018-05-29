#ifndef TRACEPATH_H
#define TRACEPATH_H



#include "curand.h"
#include "curand_kernel.h"
#include "cutil_math.h"
#include "cuda_runtime_api.h"
#include "helper_functions.h" 
#include "helper_cuda.h" 

#include "CameraCU.h"
#include "MaterialType.h"
#include <iostream>
#include <cmath>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



#define FLT_MAX_CU	3.402823466e+38F 

#define EPSILON_CU 1e-10

#define TRACE_SAMPLES 16
#define TRACE_SAMPLES_LOOP_X 6
#define TRACE_SAMPLES_LOOP_Y 6




struct TriangleCU
{
	float3 vertexes[3];
	MaterialType material;
	float3 color;
	float3 emission;
};

struct SphereCU
{
	float radius;
	float3 position;
	MaterialType material;
	float3 color;
	float3 emission;
	
};

struct ObjectIntersectionCU
{
	int hit;
	float u;
	float3 normal;
	MaterialType material;
	float3 color;
	float3 emission;
	__host__ __device__ ObjectIntersectionCU(int hit_ = 0, float u_ = 0, const float3& normal_ = make_float3(0.0f, 0.0f, 0.0f), MaterialType material_ = DIFF, const float3& color_ = make_float3(0.0f, 0.0f, 0.0f), const float3& emission_ = make_float3(0.0f, 0.0f, 0.0f))
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
	ObjectCU()
	{
		object_type = 0;
		triangles_num = 0;
		triangles_size = 0;
		triangles_p = NULL;
		sphere_p = NULL;
		//material = DIFF;
		//color = make_float3(0.0f, 0.0f, 0.0f);
		//emission = make_float3(0.0f, 0.0f, 0.0f);
	}
	int object_type;	// 0: Sphere, 1: Mesh
	unsigned int triangles_num;	// number of triangle vertexes
	unsigned int triangles_size; // byte size of triangles
	/*float3* triangles_p;
	MaterialType material;
	float3 color;
	float3 emission;*/
	TriangleCU** triangles_p;
	SphereCU* sphere_p;
};



class TracePath
{
public:
	TracePath() {}
	~TracePath(){}
	void RenderPathCUDebug(ObjectCU** object_list, int num_objects, CameraCU* camera, int* mousePos);
	float3* RenderPathCU(ObjectCU** object_list, int* num_objects, CameraCU* camera, int width, int height);
};


#endif // !TRACEPATH_CUH*/
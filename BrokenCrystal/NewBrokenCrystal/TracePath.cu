
#include "TracePath.h"

__device__ ObjectIntersectionCU IntersectCU(RayCU* ray, ObjectCU* object_list, int num_objects);
__global__ void RenderPathCUKernel(float3* output, ObjectCU* object_list, int num_objects, CameraCU* camera, int width, int height);
__device__ float3 TraceRayCU(RayCU* ray, ObjectCU* object_list, int num_objects, curandState* randState);
__device__ RayCU GetReflectedRayCU(RayCU* ray, float3 position, float3 normal, float3 color, MaterialType type, curandState* randState);



inline __host__ __device__ float clampf(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
inline __host__ __device__ float cufabs(float x)
{
	return x > 0 ? x : -x;
}


__device__ RayCU GetReflectedRayCU(RayCU* ray, float3 position, float3 normal, float3 color, MaterialType type, curandState* randState)
{
	if (type == SPEC)
	{
		float3 reflected = ray->direction - normal * 2 * dot(normal, ray->direction);
		return RayCU(position, reflected);
	}
	else if (type == GLOSS)
	{
		float roughness = 2.0f;
		float3 reflected = ray->direction - normal * 2 * dot(normal, ray->direction);
		reflected = normalize(make_float3(
			reflected.x + (curand_uniform(randState) - 0.5) * roughness,			// random generator for cuda?
			reflected.y + (curand_uniform(randState) - 0.5) * roughness,			// random generator for cuda?
			reflected.z + (curand_uniform(randState) - 0.5) * roughness			// random generator for cuda?
		));

		return RayCU(position, reflected);
	}
	else if (type == DIFF)
	{
		float3 nl = dot(normal, ray->direction) < 0 ? normal : normal * -1;
		float r1 = 2 * CU_SIMD_PI * curand_uniform(randState);						// random generator for cuda?
		float r2 = curand_uniform(randState);									// random generator for cuda?
		float r2s = sqrt(r2);

		float3 w = nl;
		float3 u;
		if (cufabs(w.x) > 0.1)
			u = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), w));
		else
			u = normalize(cross(make_float3(1.0f, 0.0f, 0.0f), w));
		float3 v = cross(w, u);
		float3 d = normalize((u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)));
		return RayCU(position, d);
	}
	else if (type == TRANS)
	{
		float3 nl = dot(normal, ray->direction) < 0 ? normal : normal * -1;
		float3 reflection = ray->direction - normal * 2 * dot(normal, ray->direction);
		bool into = dot(normal, nl) > 0;
		float nc = 1.0f;
		float nt = 1.5f;
		float nnt;

		float Re, RP, TP, Tr;
		float3 tdir = make_float3(0.0f, 0.0f, 0.0f);

		if (into)
			nnt = nc / nt;
		else
			nnt = nt / nc;

		float ddn = dot(ray->direction, nl);
		float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);

		if (cos2t < 0) return RayCU(position, reflection);

		if (into)
			tdir = normalize((ray->direction * nnt - normal * (ddn * nnt + sqrt(cos2t))));
		else
			tdir = normalize((ray->direction * nnt + normal * (ddn * nnt + sqrt(cos2t))));

		float a = nt - nc;
		float b = nt + nc;
		float R0 = a * a / (b * b);

		float c;
		if (into)
			c = 1 + ddn;
		else
			c = 1 - dot(tdir, normal);

		Re = R0 + (1 - R0) * c * c * c * c * c;
		Tr = 1 - Re;

		float P = .25 + .5 * Re;
		RP = Re / P;
		TP = Tr / (1 - P);

		if (curand_uniform(randState) < P)		// random generator for cuda?
		{
			color = color * (RP);
			return RayCU(position, reflection);
		}

		color = color * (TP);
		return RayCU(position, tdir);
	}
	else {}
}


__global__ void initCURand(unsigned int seed, curandState_t* states)
{

	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
		blockIdx.x, /* the sequence number should be different for each core (unless you want all
					cores to get the same sequence of numbers for some reason - use thread id! */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&states[blockIdx.x]);
}



float3* TracePath::RenderPathCU(ObjectCU* object_list, int num_objects, CameraCU* camera, int width, int height) // need to use camera not CameraCU but Camera in Scene
{
	std::cout << "RenderPathCU entered successfully" << std::endl;
	float3* output_host = new float3[width * height];
	float3* output_device;

//	curandstate_t* states;
//	cudaMalloc((void**)&states, THREAD_NUM * sizeof(curandState_t));

	cudaMalloc(&output_device, width * height * sizeof(float3));

	std::cout << "output_device cudaMalloc successed" << std::endl;

	dim3 block(16, 16, 1); // calculate this

	size_t blocks_width = ceilf(width / block.x);
	size_t blocks_height = ceilf(height / block.y);

	dim3 grid(blocks_width, blocks_height, 1);  // calculate this

	std::cout << "dim set successed" << std::endl;

	// cuda 내부 디버그 memcpy 이용하자

	RenderPathCUKernel <<< grid, block >>>(output_device, object_list, num_objects, camera, width, height);

	std::cout << "RenderPathCUKernel successed" << std::endl;

	cudaMemcpy(output_host, output_device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

	std::cout << "copy result device to host successed" << std::endl;

	cudaFree(output_device);
	cudaFree(object_list);
	cudaFree(camera);

	std::cout << "cudaFree successed" << std::endl;

	return output_host;
}

__global__ void RenderPathCUKernel(float3* output, ObjectCU* object_list, int num_objects, CameraCU* camera, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curandState randState;
	curand_init(threadId, 0, 0, &randState);

	int pixelCount = width * height;
	for (int t = 0; t < pixelCount; t++)
	{
		int x = t % width;
		int y = t / width;

		float samplesP = 1.0f / TRACE_SAMPLES;
		float3 resultcolor = make_float3(0.0f, 0.0f, 0.0f);

		for (int sy = 0; sy < 2; sy++)
		{
			for (int sx = 0; sx < 2; sx++)
			{
				float3 color = make_float3(0.0f, 0.0f, 0.0f);
				for (int s = 0; s < TRACE_SAMPLES; s++)
				{
					RayCU ray = camera->GetRay(&randState, x, y, sx, sy, 0); // ***
					color = color + TraceRayCU(&ray, object_list, num_objects, &randState);
				}

				resultcolor = resultcolor + color * samplesP;
			}
		}

		resultcolor = resultcolor * 0.25f;
		output[t] = make_float3(clampf(resultcolor.x), clampf(resultcolor.y), clampf(resultcolor.z));
	}
}

__device__ float3 TraceRayCU(RayCU* ray, ObjectCU* object_list, int num_objects, curandState* randState)
{
	float3 result_color = make_float3(0.0f, 0.0f, 0.0f);

	for (int depth = 0; depth < 15; depth++)
	{
		ObjectIntersectionCU intersection = IntersectCU(ray, object_list, num_objects);
		if (!intersection.hit) return make_float3(0.0f, 0.0f, 0.0f);
		if (intersection.material == EMIT) return intersection.emission;	// need to be fixed

		float3 color = intersection.color;
		float maxReflection = ((color.x > color.y && color.x > color.z) ? color.x : (color.y > color.z)) ? color.y : color.z;
		float random = curand_uniform(randState);// random number generator for cuda?
		
		if (random < maxReflection * 0.9)
		{
			color = color * (0.9 / maxReflection);
		}
		else
		{
			return result_color * intersection.emission;
		}

		if (depth == 0)
		{
			result_color = color;
		}
		else
		{
			result_color = result_color * color;
		}

		float3 pos = ray->origin + ray->direction * intersection.u;
		RayCU reflected = GetReflectedRayCU(ray, pos, intersection.normal, color, intersection.material, randState);
	}
	return result_color;
}




__device__ ObjectIntersectionCU IntersectCU(RayCU* ray, ObjectCU* object_list, int num_objects)
{
	ObjectIntersectionCU intersection = ObjectIntersectionCU();
	ObjectIntersectionCU temp = ObjectIntersectionCU();	// return value of objects.at((unsigned)i)->GetIntersection(ray)
	ObjectCU current_obj;

	ObjectIntersectionCU temp_inner = ObjectIntersectionCU(); // return value of triangle->GetIntersect()

	for (int i = 0; i < num_objects; i++)
	{
		current_obj = object_list[i];

		float tNear = FLT_MAX_CU;

		for (unsigned int i = 0; i < current_obj.triangles_size; i += 3)
		{
			float3 v0 = current_obj.triangles_p[i];
			float3 v1 = current_obj.triangles_p[i + 1];
			float3 v2 = current_obj.triangles_p[i + 2];

			// triangle->GetIntersection(ray, transform)

			bool hit = false;
			float u, v, t = 0;

			float3 normal = normalize(cross(v1 - v0, v2 - v0));

			float3 v0v1 = v1 - v0;
			float3 v0v2 = v2 - v0;
			float3 pvec = cross(ray->direction, v0v2);
			float det = dot(v0v1, pvec);
			if (cufabs(det) < EPSILON_CU)
			{
				temp_inner.hit = hit;
				temp_inner.material = MaterialType(current_obj.material);
				temp_inner.u = t;
				temp_inner.normal = normal;
				if (temp_inner.hit && temp_inner.u < tNear)
				{
					tNear = temp_inner.u;
					temp.hit = temp_inner.hit;
					temp.material = temp_inner.material;
					temp.normal = temp_inner.normal;
					temp.u = temp_inner.u;
				}
				continue;
			}

			float3 tvec = ray->origin - v0;
			u = dot(tvec, pvec);
			if (u < 0 || u > det)
			{
				temp_inner.hit = hit;
				temp_inner.material = MaterialType(current_obj.material);
				temp_inner.u = t;
				temp_inner.normal = normal;
				if (temp_inner.hit && temp_inner.u < tNear)
				{
					tNear = temp_inner.u;
					temp.hit = temp_inner.hit;
					temp.material = temp_inner.material;
					temp.normal = temp_inner.normal;
					temp.u = temp_inner.u;
				}
				continue;
			}

			float3 qvec = cross(tvec, v0v1);
			v = dot(ray->direction, qvec);
			if (v < 0 || u + v > det)
			{
				temp_inner.hit = hit;
				temp_inner.material = MaterialType(current_obj.material);
				temp_inner.u = t;
				temp_inner.normal = normal;
				if (temp_inner.hit && temp_inner.u < tNear)
				{
					tNear = temp_inner.u;
					temp.hit = temp_inner.hit;
					temp.material = temp_inner.material;
					temp.normal = temp_inner.normal;
					temp.u = temp_inner.u;
				}
				continue;
			}

			t = dot(v0v2, qvec) / det;

			if (t < EPSILON_CU)
			{
				temp_inner.hit = hit;
				temp_inner.material = MaterialType(current_obj.material);
				temp_inner.u = t;
				temp_inner.normal = normal;
				if (temp_inner.hit && temp_inner.u < tNear)
				{
					tNear = temp_inner.u;
					temp.hit = temp_inner.hit;
					temp.material = temp_inner.material;
					temp.normal = temp_inner.normal;
					temp.u = temp_inner.u;
				}
				continue;
			}

			hit = true;

			temp_inner.hit = hit;
			temp_inner.material = MaterialType(current_obj.material);
			temp_inner.u = t;
			temp_inner.normal = normal;
			if (temp_inner.hit && temp_inner.u < tNear)
			{
				tNear = temp_inner.u;
				temp.hit = temp_inner.hit;
				temp.material = temp_inner.material;
				temp.normal = temp_inner.normal;
				temp.u = temp_inner.u;
			}
		}

		if (temp.hit)
		{
			if (intersection.u == 0 || temp.u < intersection.u)
			{
				intersection.hit = temp.hit;
				intersection.material = temp.material;
				intersection.normal = temp.normal;
				intersection.u = temp.u;
				intersection.color = current_obj.color;
				intersection.emission = current_obj.emission;
			}
		}
	}

	return intersection;
}

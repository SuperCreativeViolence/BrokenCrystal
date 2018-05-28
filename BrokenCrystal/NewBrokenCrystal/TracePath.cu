
#include "TracePath.h"

__global__ void RenderPathCUDebugKernel(ObjectIntersectionCU* output, ObjectCU** object_list, int num_objects, CameraCU* camera, int* mousePos);
__device__ ObjectIntersectionCU IntersectCU(RayCU* ray, ObjectCU** object_list, int* num_objects, float3* debug_buffer, int thread_index);
__global__ void RenderPathCUKernelLoop(float3* output, ObjectCU** object_list, int* num_objects, CameraCU* camera, float3* debug_buffer, int* loop_x, int* loop_y);
__global__ void RenderPathCUKernel(float3* output, ObjectCU** object_list, int* num_objects, CameraCU* camera, float3* debug_buffer);
__device__ float3 TraceRayCU(RayCU* ray, ObjectCU** object_list, int* num_objects, curandState* randState, float3* debug_buffer, int thread_index);
__device__ RayCU GetReflectedRayCU(RayCU* ray, float3 position, float3 normal, float3 color, MaterialType type, curandState* randState);



inline __host__ __device__ float clampf(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
inline __host__ __device__ float cufabs(float x)
{
	return x > 0 ? x : -x;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
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
		float roughness = 0.5f;
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
		float3 nl = dot(normal, ray->direction) < 0.0f ? normal : normal * -1.0f;
		float r1 = 2.0f * CU_SIMD_PI * curand_uniform(randState);						// random generator for cuda?
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

void TracePath::RenderPathCUDebug(ObjectCU** object_list, int num_objects, CameraCU* camera, int* mousePos)
{
	ObjectIntersectionCU* output_host = new ObjectIntersectionCU;
	ObjectIntersectionCU* output_device;

	int* mousePos_device;

	cudaMalloc((void**)&output_device, sizeof(ObjectIntersectionCU));
	cudaMalloc((void**)&mousePos_device, sizeof(int) * 2);

	cudaMemcpy(mousePos_device, mousePos, sizeof(int) * 2, cudaMemcpyHostToDevice);

	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);


	RenderPathCUDebugKernel << <grid, block >> > (output_device, object_list, num_objects, camera, mousePos_device);

	cudaMemcpy(output_host, output_device, sizeof(ObjectIntersectionCU), cudaMemcpyDeviceToHost);

	if (output_host[0].hit == 0)
	{
		std::cout << "No Hit!" << std::endl;
	}
	else if (output_host[0].material == EMIT)
	{
		std::cout << "EMIT!" << std::endl;
	}
	else
	{
		printf("Hit : %f | normal :  %.1f %.1f %.1f | color : %.1f %.1f %.1f\n", output_host[0].u, output_host[0].normal.x, output_host[0].normal.y, output_host[0].normal.z, output_host[0].color.x, output_host[0].color.y, output_host[0].color.z);
	}


	cudaFree(output_device);
	cudaFree(object_list);
	cudaFree(camera);
	cudaFree(mousePos_device);
	delete output_host;
}

__global__ void RenderPathCUDebugKernel(ObjectIntersectionCU* output, ObjectCU** object_list, int num_objects, CameraCU* camera, int* mousePos)
{
	curandState randState;
	curand_init(0, 0, 0, &randState);
	RayCU ray = camera->GetRay(&randState, mousePos[0], mousePos[1], 0, 0, 0);
	ObjectIntersectionCU intersection = ObjectIntersectionCU();
	ObjectIntersectionCU temp = ObjectIntersectionCU();	// return value of objects.at((unsigned)i)->GetIntersection(ray)
	ObjectCU* current_obj;

	ObjectIntersectionCU temp_inner = ObjectIntersectionCU(); // return value of triangle->GetIntersect()

	for (int i = 0; i < num_objects; i++)
	{
		current_obj = object_list[i];

		float tNear = FLT_MAX_CU;

		for (unsigned int j = 0; j < current_obj->triangles_num; j += 3)
		{
			float3 v0 = current_obj->triangles_p[j];
			float3 v1 = current_obj->triangles_p[j + 1];
			float3 v2 = current_obj->triangles_p[j + 2];

			// triangle->GetIntersection(ray, transform)

			int hit = 0;
			float u, v, t = 0;

			float3 normal = normalize(cross(v1 - v0, v2 - v0));

			float3 v0v1 = v1 - v0;
			float3 v0v2 = v2 - v0;
			float3 pvec = cross(ray.direction, v0v2);
			float det = dot(v0v1, pvec);

			if (cufabs(det) < EPSILON_CU)
			{

				temp_inner.hit = hit;
				temp_inner.material = current_obj->material;
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

			float3 tvec = ray.origin - v0;
			u = dot(tvec, pvec);

			if (u < 0 || u > det)
			{

				temp_inner.hit = hit;
				temp_inner.material = current_obj->material;
				temp_inner.u = t;
				temp_inner.normal = normal;
				if (temp_inner.hit == 1 && temp_inner.u < tNear)
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
			v = dot(ray.direction, qvec);

			if (v < 0 || u + v > det)
			{
				temp_inner.hit = hit;
				temp_inner.material = current_obj->material;
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
				temp_inner.material = current_obj->material;
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

			hit = 1;

			temp_inner.hit = hit;
			temp_inner.material = current_obj->material;
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

		if (temp.hit == 1)
		{
			if (intersection.u == 0 || temp.u < intersection.u)
			{
				intersection.hit = temp.hit;
				intersection.material = temp.material;
				intersection.normal = temp.normal;
				intersection.u = temp.u;
				intersection.color = current_obj->color;
				intersection.emission = current_obj->emission;
			}
		}
	}
	output->color = intersection.color;
	output->emission = intersection.emission;
	output->hit = intersection.hit;
	output->material = intersection.material;
	output->normal = intersection.normal;
	output->u = intersection.u;
}



float3* TracePath::RenderPathCU(ObjectCU** object_list, int* num_objects, CameraCU* camera, int width, int height) // need to use camera not CameraCU but Camera in Scene
{
	// time
	unsigned int startTime = time(nullptr);

	std::cout << "RenderPathCU entered successfully" << std::endl;
	float3* output_host = new float3[width * height];
	float3* output_device;

	float3* debug_host = new float3[width * height];
	float3* debug_device;


	cudaMalloc(&output_device, width * height * sizeof(float3));

	cudaMalloc(&debug_device, width * height * sizeof(float3));

	std::cout << "output_device cudaMalloc successed" << std::endl;

	dim3 block(16, 9, 1); // calculate this

	size_t blocks_width = ceilf(width / block.x);
	size_t blocks_height = ceilf(height / block.y);

	dim3 grid(20, 20, 1);  // calculate this
	//dim3 grid(blocks_width, blocks_height, 1);

	std::cout << "dim set successed" << std::endl;
	std::cout << "block : " << block.x << block.y << block.z << " grid : " << grid.x << grid.y << grid.z << std::endl;

	// loop mem allocate
	int* loop_x_device;
	int* loop_y_device;
	int loop_x_host = 0;
	int loop_y_host = 0;
	cudaMalloc((void**)&loop_x_device, sizeof(int));
	cudaMalloc((void**)&loop_y_device, sizeof(int));


	// loop version
	for (loop_x_host = 0; loop_x_host < TRACE_SAMPLES_LOOP_X; loop_x_host++)
	{
		for (loop_y_host = 0; loop_y_host < TRACE_SAMPLES_LOOP_Y; loop_y_host++)
		{
			cudaMemcpy(loop_x_device, &loop_x_host, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(loop_y_device, &loop_y_host, sizeof(int), cudaMemcpyHostToDevice);

			RenderPathCUKernelLoop << < grid, block >> > (output_device, object_list, num_objects, camera, debug_device, loop_x_device, loop_y_device);
			
			cudaError_t error = cudaDeviceSynchronize();
			if (error != cudaSuccess)
			{
				// print the CUDA error message and exit
				printf("CUDA error: %s\n", cudaGetErrorString(error));
			}
			std::cout << "Current loop: " << loop_x_host << " " << loop_y_host << std::endl;
			//Sleep(1000);
		}
	}

	// no loop version

	//RenderPathCUKernel << < grid, block >> > (output_device, object_list, num_objects, camera, debug_device);
	//Sleep(100);
	/*
	cudaError_t error = cudaDeviceSynchronize();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}*/



	
	// make the host block until the device is finished with foo

	// check for error
	/*error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}*/

	std::cout << "RenderPathCUKernel successed" << std::endl;

	cudaMemcpy(output_host, output_device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

	cudaMemcpy(debug_host, debug_device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

	std::cout << "copy result device to host successed" << std::endl;

	/* debug */
	/*for (int i = 0; i < width * height; i++)
	{
		//std::cout << debug_host[i].x << " " << debug_host[i].y << " " << debug_host[i].z << std::endl;
		//if(debug_host[i].x == 4.0f)
		//std::cout << debug_host[i].x <<" "<<debug_host[i].y << " " << debug_host[i].z << std::endl;
	}*/
	/* debug end */
	/*	
	for (int i = 0; i < num_objects; i++)
	{
		cudaFree(object_list[i]->triangles_p);
		cudaFree(object_list);
	}*/
	
	// check for error

	cudaFree(output_device);
	cudaFree(object_list);
	cudaFree(camera);
	cudaFree(num_objects);
	cudaFree(loop_x_device);
	cudaFree(loop_y_device);
	cudaFree(debug_device);
	delete debug_host;

	std::cout << "cudaFree successed" << std::endl;

	int elapsedTime = (int)difftime(time(nullptr), startTime);
	printf("\rCUDA PathTracing complete, time taken: %.2dh%.2dm%.2ds.\n", elapsedTime / 3600, (elapsedTime % 3600) / 60, elapsedTime % 60);

	return output_host;
}

__global__ void RenderPathCUKernel(float3* output, ObjectCU** object_list, int* num_objects, CameraCU* camera, float3* debug_buffer)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curandState randState;
	curand_init(threadId, 0, 0, &randState);
	int i = y * camera->width + x;


	if (i < camera->width * camera->height)
	{
		//debug_buffer[i] = make_float3(0.0f, 0.0f, 0.0f);

		float samplesP = 1.0f / TRACE_SAMPLES;
		float3 resultcolor = make_float3(0.0f, 0.0f, 0.0f);

		//debug_buffer[i].x = curand_uniform(&randState);
		//debug_buffer[i].y = curand_uniform(&randState);

		for (int sy = 0; sy < 2; sy++)
		{
			for (int sx = 0; sx < 2; sx++)
			{
				float3 color = make_float3(0.0f, 0.0f, 0.0f);
				for (int s = 0; s < TRACE_SAMPLES; s++)
				{
					RayCU ray = camera->GetRay(&randState, x, y, sx, sy, 0); // ***
																			 //ray = camera->GetRay(&randState, x, y, sx, sy, 0); // ***
					color = color + TraceRayCU(&ray, object_list, num_objects, &randState, debug_buffer, i);
					//free(&ray);

				}

				resultcolor = resultcolor + (color * samplesP);
			}
		}

		resultcolor = resultcolor * 0.25f;
		output[i] = make_float3(clampf(resultcolor.x), clampf(resultcolor.y), clampf(resultcolor.z));

		free(&randState);
		/* debug */
		//debug_buffer[i].x = i;
		/* debug end */
	}
}

__global__ void RenderPathCUKernelLoop(float3* output, ObjectCU** object_list, int* num_objects, CameraCU* camera, float3* debug_buffer, int* loop_x, int* loop_y)
{
	int x = gridDim.x * blockDim.x * loop_x[0] + blockIdx.x * blockDim.x + threadIdx.x;
	int y = gridDim.y * blockDim.y * loop_y[0] + blockIdx.y * blockDim.y + threadIdx.y;

	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curandState randState;
	curand_init(threadId, 0, 0, &randState);
	int i = y * camera->width + x;


	if (i < camera->width * camera->height)
	{
		//debug_buffer[i] = make_float3(0.0f, 0.0f, 0.0f);
		
		float samplesP = 1.0f / TRACE_SAMPLES;
		float3 resultcolor = make_float3(0.0f, 0.0f, 0.0f);

		//debug_buffer[i].x = curand_uniform(&randState);
		//debug_buffer[i].y = curand_uniform(&randState);

		for (int sy = 0; sy < 2; sy++)
		{
			for (int sx = 0; sx < 2; sx++)
			{
				float3 color = make_float3(0.0f, 0.0f, 0.0f);
				for (int s = 0; s < TRACE_SAMPLES; s++)
				{
					RayCU ray = camera->GetRay(&randState, x, y, sx, sy, 0); // ***
					//ray = camera->GetRay(&randState, x, y, sx, sy, 0); // ***
					color = color + TraceRayCU(&ray, object_list, num_objects, &randState, debug_buffer, i);
					//free(&ray);

				}

				resultcolor = resultcolor + (color * samplesP);
			}
		}

		resultcolor = resultcolor * 0.25f;
		output[i] = make_float3(clampf(resultcolor.x), clampf(resultcolor.y), clampf(resultcolor.z));
		
		free(&randState);
		/* debug */
		//debug_buffer[i].x = i;
		/* debug end */
	}
}

__device__ float3 TraceRayCU(RayCU* ray, ObjectCU** object_list, int* num_objects, curandState* randState, float3* debug_buffer, int thread_index)
{
	float3 result_color = make_float3(.0f, .0f, .0f);
	
	for (int depth = 0; depth < 15; depth++)
	{
		ObjectIntersectionCU intersection = IntersectCU(ray, object_list, num_objects, debug_buffer, thread_index);
		// debug 
		//debug_buffer[thread_index].x = (int)intersection.material;

		// debug end
		
		if (intersection.hit == 0) return make_float3(.0f, .0f, .0f);
		if (intersection.material == EMIT)
		{
			//debug_buffer[thread_index].x += 1.0f;
			if (depth == 0)
			{
				return intersection.emission;
			}
			else
			{
				return result_color * intersection.emission;
			}
		}	// need to be fixed

		float3 color = intersection.color;
		float maxReflection = color.x > color.y && color.x > color.z ? color.x : color.y > color.z ? color.y : color.z;
		float random = curand_uniform(randState);// random number generator for cuda?

		if (depth > 5)
		{
			if (random < maxReflection * 0.9f)
			{
				color = color * (0.9f / maxReflection);
			}
			else
			{
				return result_color * intersection.emission;
			}
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
		ray = &reflected;
	}
	return result_color;

}


__device__ ObjectIntersectionCU IntersectCU(RayCU* ray, ObjectCU** object_list, int* num_objects, float3* debug_buffer, int thread_index)
{
	ObjectIntersectionCU intersection = ObjectIntersectionCU();
	ObjectIntersectionCU temp = ObjectIntersectionCU();	// return value of objects.at((unsigned)i)->GetIntersection(ray)
	ObjectCU* current_obj;

	ObjectIntersectionCU temp_inner = ObjectIntersectionCU(); // return value of triangle->GetIntersect()

	for (int i = 0; i < num_objects[0]; i++)
	{
		current_obj = object_list[i];

		float tNear = FLT_MAX_CU;

		//debug_buffer[thread_index].z = num_objects;
		for (unsigned int j = 0; j < current_obj->triangles_num; j += 3)
		{
		
			//debug_buffer[thread_index].z += 1.0f;


			float3 v0 = current_obj->triangles_p[j];
			float3 v1 = current_obj->triangles_p[j + 1];
			float3 v2 = current_obj->triangles_p[j + 2];
			
			// triangle->GetIntersection(ray, transform)

			int hit = 0;
			float u, v, t = 0;

			float3 normal = normalize(cross(v1 - v0, v2 - v0));

			float3 v0v1 = v1 - v0;
			float3 v0v2 = v2 - v0;
			float3 pvec = cross(ray->direction, v0v2);
			float det = dot(v0v1, pvec);

			if (cufabs(det) < EPSILON_CU)
			{

				temp_inner.hit = hit;
				temp_inner.material = current_obj->material;
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
				temp_inner.material = current_obj->material;
				temp_inner.u = t;
				temp_inner.normal = normal;
				if (temp_inner.hit == 1 && temp_inner.u < tNear)
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

				//debug_buffer[thread_index].z += 1.0f;
				temp_inner.hit = hit;
				temp_inner.material = current_obj->material;
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

				//debug_buffer[thread_index].x += 1.0f;
				temp_inner.hit = hit;
				temp_inner.material = current_obj->material;
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

			hit = 1;

			//debug_buffer[thread_index].y += 1.0f;

			temp_inner.hit = hit;
			temp_inner.material = current_obj->material;
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
		
		if (temp.hit == 1)
		{

			//if (temp.material == EMIT) debug_buffer[thread_index].x += 1.0f;
			if (intersection.u == 0 || temp.u < intersection.u)
			{
				//debug_buffer[thread_index].y += 1.0f;
				intersection.hit = temp.hit;
				intersection.material = temp.material;
				intersection.normal = temp.normal;
				intersection.u = temp.u;
				intersection.color = current_obj->color;
				intersection.emission = current_obj->emission;
			}
		}
	}
	/*
	free(&temp);
	free(current_obj);
	free(&temp_inner);*/

	return intersection;
}

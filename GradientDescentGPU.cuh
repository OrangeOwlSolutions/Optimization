#ifndef GRADIENT_DESCENT_GPU
#define GRADIENT_DESCENT_GPU

void GradientDescentGPU(const float * __restrict__, const float, const int, const float, const float, const float, const int, 
	                          float * __restrict__, float *, int *, float *, float *);

__host__ __device__ float CostFunction(const float * __restrict, const int);

#endif


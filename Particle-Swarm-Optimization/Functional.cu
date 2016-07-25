#include <cuda.h>
#include <cuda_runtime.h>

/************************/
/* RASTRIGIN FUNCTIONAL */
/************************/
__device__ float rastrigin(float x) { return x * x - 10.0f * cosf(2.0f * x) + 10.0f; }

/*********************/
/* SPHERE FUNCTIONAL */
/*********************/
__device__ float sphere(float x) { return (x - 0.5f) * (x - 0.5f); }


#include <thrust\device_vector.h>
#include <thrust\inner_product.h>

#include "Utilities.cuh"
#include "FuncGrad.cuh"
#include "CostFunctionStructGPU.cuh"

/*****************/
/* SAXPY FUNCTOR */
/*****************/
struct saxpy_functor : public thrust::binary_function<float, float, float>
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__ float operator()(const float& x, const float& y) const { 
		return x + a * y; }
};

/*******************************/
/* GRADIENT DESCENT - GPU CASE */
/*******************************/
// --- Version using analytical gradient (Rosenbrock function)
__device__ void CostFunctionGradientGPU(const float * __restrict d_x, float * __restrict d_g, const int tid, const int N) {

	if (N > 2) {
		if (tid == 0) d_g[0] = -400.f * (d_x[1] - d_x[0] * d_x[0]) * d_x[0] + 2.f * (d_x[0] - 1.f);
		else if (tid == N-1) d_g[N-1] = 200.f * (d_x[N-1] - d_x[N-2] * d_x[N-2]);
		else {
			d_g[tid]	= -400.f * d_x[tid] * (d_x[tid+1] - d_x[tid] * d_x[tid]) + 2.f * (d_x[tid] - 1.f) + 200.f * (d_x[tid] - d_x[tid-1] * d_x[tid-1]);
		}
	}
	else {
		if (tid == 0) d_g[0] = -400.f * (d_x[1] - d_x[0] * d_x[0]) * d_x[0] + 2.f * (d_x[0] - 1.f);	
		else	      d_g[1] =  200.f * (d_x[1] - d_x[0] * d_x[0]);	
	}

}

/************************/
/* GRADIENT CALCULATION */
/************************/
__global__ void GradientCalculation(const float * __restrict d_x, float * __restrict d_g, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid < N)

		// --- Calculate gradient
		CostFunctionGradientGPU(d_x, d_g, tid, N);

}

/******************/
/* F1DIM FUNCTION */
/******************/
float f1dim(const thrust::device_ptr<float> dev_ptr_x, const thrust::device_ptr<float> dev_ptr_p, const thrust::device_ptr<float> dev_ptr_xt, const float lambda, const int N) {

	thrust::transform(dev_ptr_x, dev_ptr_x + N, dev_ptr_p, dev_ptr_xt, saxpy_functor(lambda));
	
	return thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_xt, dev_ptr_xt + 1)), thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_xt + N - 1, dev_ptr_xt + N)), CostFunctionStructGPU(), 0.0f, thrust::plus<float>());

}

/*******************/
/* DF1DIM FUNCTION */
/*******************/
float df1dim(float *d_xt, const thrust::device_ptr<float> dev_ptr_p, const float lambda, const int N) {

	float *d_Grad; gpuErrchk(cudaMalloc((void**)&d_Grad, N * sizeof(float)));			thrust::device_ptr<float> dev_ptr_Grad			= thrust::device_pointer_cast(d_Grad);	
	GradientCalculation<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_xt, d_Grad, N);

	return thrust::inner_product(dev_ptr_Grad,		dev_ptr_Grad + N,		dev_ptr_p,		0.0f);

}

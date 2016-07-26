#include <thrust\device_ptr.h>
#include <thrust\inner_product.h>

#include "Utilities.cuh"

#define BLOCK_SIZE 256

//#define VERBOSE
//#define DEBUG

/***********************************/
/* COST FUNCTION - CPU & GPU CASES */
/***********************************/
__host__ __device__ float CostFunction(const float * __restrict h_x, const int M) {

	// --- Rosenbrock function
	float sum = 0.f;
	for (int i=0; i<M-1; i++) {
		float temp1 = (h_x[i+1] - h_x[i] * h_x[i]);
		float temp2 = (h_x[i] - 1.f);
		sum = sum + 100.f * temp1 * temp1 + temp2 * temp2;
	}
	return sum;
}

/*******************************/
/* GRADIENT DESCENT - GPU CASE */
/*******************************/

// --- Version using finite differences
//__device__ void CostFunctionGradientGPU(float * __restrict d_x, float * __restrict d_g, const float h, const int tid, const int M) {
//
//	int test1, test2;
//	float h_test1_plus, h_test1_minus, h_test2_plus, h_test2_minus, temp1_plus, temp1_minus, temp2_plus, temp2_minus;
//	
//	// --- Rosenbrock function
//	float sum_plus = 0.f, sum_minus = 0.f;
//	for (int i=0; i<M-1; i++) {
//		h_test1_plus	= d_x[i] + (h / 2.f) * (tid == i);
//		h_test1_minus	= d_x[i] - (h / 2.f) * (tid == i);
//		h_test2_plus	= d_x[i + 1] + (h / 2.f) * (tid == (i + 1));
//		h_test2_minus	= d_x[i + 1] - (h / 2.f) * (tid == (i + 1));
//		temp1_plus		= (h_test2_plus - h_test1_plus * h_test1_plus);
//		temp2_plus		= (h_test1_plus - 1.f);
//		temp1_minus		= (h_test2_minus - h_test1_minus * h_test1_minus);
//		temp2_minus		= (h_test1_minus - 1.f);
//		sum_plus		= sum_plus  + 100.f * temp1_plus  * temp1_plus  + temp2_plus  * temp2_plus;
//		sum_minus		= sum_minus + 100.f * temp1_minus * temp1_minus + temp2_minus * temp2_minus;
//	}
//	d_g[tid] = (sum_plus - sum_minus) / (2.f * h);
//}

// --- Version using analytical gradient (Rosenbrock function)
__device__ void CostFunctionGradientGPU(float * __restrict d_x, float * __restrict d_g, const float h, const int tid, const int M) {

	if (tid == 0) d_g[0] = -400.f * (d_x[1] - d_x[0] * d_x[0]) * d_x[0] + 2.f * (d_x[0] - 1.f);
	else if (tid == M-1) d_g[M-1] = 200.f * (d_x[M-1] - d_x[M-2] * d_x[M-2]);
	else {
		for (int i=1; i<M-1; i++) {
			d_g[i]	= -400.f * d_x[i] * (d_x[i+1] - d_x[i] * d_x[i]) + 2.f * (d_x[i] - 1.f) + 200.f * (d_x[i] - d_x[i-1] * d_x[i-1]);
		}
	}
}

/*******************/
/* STEP - GPU CASE */
/*******************/
__global__ void StepGPU(float * __restrict d_x, float * __restrict d_xnew, float * __restrict d_xdiff, float * __restrict d_g, const float alpha, const float h, const int M) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid < M) {
	
		// --- Calculate gradient
		CostFunctionGradientGPU(d_x, d_g, h, tid, M);

	    // --- Take step
		d_xnew[tid] = d_x[tid] - alpha * d_g[tid];

		// --- Update termination metrics
		d_xdiff[tid] = d_xnew[tid] - d_x[tid];

		// --- Update current solution
		d_x[tid] = d_xnew[tid];
	}

}

/***********************************/
/* COST FUNCTION STRUCT - GPU CASE */
/***********************************/

// --- Rosenbrock function struct for thrust reduction
struct CostFunctionStructGPU{
template <typename Tuple>
	__host__ __device__ float operator()(Tuple a) {

		float temp1 = (thrust::get<1>(a) - thrust::get<0>(a) * thrust::get<0>(a));
		float temp2 = (thrust::get<0>(a) - 1.f);
	
		return 100.f * temp1 * temp1 + temp2 * temp2;
	}
};


/****************************************/
/* GRADIENT DESCENT FUNCTION - GPU CASE */
/****************************************/

// x0      - Starting point
// tol     - Termination tolerance
// maxiter - Maximum number of allowed iterations
// alpha   - Step size
// dxmin   - Minimum allowed perturbations

void GradientDescentGPU(const float * __restrict__ d_x0, const float tol, const int maxiter, const float alpha, const float h, 
						const float dxmin, const int M, float * __restrict__ d_xopt, float *fopt, int *niter, float *gnorm, float *dx) {

	thrust::device_ptr<float> dev_ptr_xopt		= thrust::device_pointer_cast(d_xopt);	
								  
	// --- Initialize gradient norm, optimization vector, iteration counter, perturbation
	*gnorm = FLT_MAX; 
	
	float *d_x;			gpuErrchk(cudaMalloc((void**)&d_x, M * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_x, d_x0, M * sizeof(float), cudaMemcpyDeviceToDevice));

	*niter = 0;

	*dx = FLT_MAX;
		
	// --- Allocating space for the gradient, for the new actual solution and for the difference between actual and old solutions
	float *d_g;			gpuErrchk(cudaMalloc((void**)&d_g, M * sizeof(float)));			thrust::device_ptr<float> dev_ptr_g		= thrust::device_pointer_cast(d_g);
	float *d_xnew;		gpuErrchk(cudaMalloc((void**)&d_xnew, M * sizeof(float)));		
	float *d_xdiff;		gpuErrchk(cudaMalloc((void**)&d_xdiff, M * sizeof(float)));		thrust::device_ptr<float> dev_ptr_xdiff	= thrust::device_pointer_cast(d_xdiff);
	
	// --- Gradient Descent iterations
	while ((*gnorm >= tol) && (*niter <= maxiter) && (*dx >= dxmin)) {
		
	    // --- Iteration step
		StepGPU<<<iDivUp(M, BLOCK_SIZE), BLOCK_SIZE>>>(d_x, d_xnew, d_xdiff, d_g, alpha, h, M);
#ifdef DEBUG
	    gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		*gnorm	= sqrt(thrust::inner_product(dev_ptr_g,		dev_ptr_g + M,		dev_ptr_g,		0.0f));
		*dx		= sqrt(thrust::inner_product(dev_ptr_xdiff,	dev_ptr_xdiff + M,	dev_ptr_xdiff,	0.0f));
		*niter	= *niter + 1;

	}

	gpuErrchk(cudaMemcpy(d_xopt, d_x, M * sizeof(float), cudaMemcpyDeviceToDevice));

	// --- Functional calculation
	*fopt = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_xopt, dev_ptr_xopt + 1)), thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_xopt + M - 1, dev_ptr_xopt + M)), CostFunctionStructGPU(), 0.0f, thrust::plus<float>());

	*niter = *niter - 1;

}

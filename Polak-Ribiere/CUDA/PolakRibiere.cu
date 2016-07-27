#include <thrust\device_vector.h>
#include <thrust\inner_product.h>

#include "CostFunctionStructGPU.cuh"
#include "FuncGrad.cuh"
#include "linmin.cuh"
#include "Utilities.cuh"

/*****************/
/* SAXPY FUNCTOR */
/*****************/
struct PR_saxpy_functor : public thrust::binary_function<float, float, float>
{
    const float a;

    PR_saxpy_functor(float _a) : a(_a) {}

    __host__ __device__ float operator()(const float& x, const float& y) const { 
		return a * x - y; }
};

/******************************************/
/* INITIALIZATION OF THE SEARCH DIRECTION */
/******************************************/
__global__ void InitXI(float * __restrict d_XI, float * __restrict d_Grad, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid < N)

		// --- Calculate gradient
		d_XI[tid] = - d_Grad[tid];

}

/**************************************/
/* POLAK-RIBIERE MINIMIZATION ROUTINE */
/**************************************/
void PolakRibiere(const float * __restrict h_xstart, float * __restrict h_xfinal, const int itmax, const int itmaxlinmin, const float Ftol, const int N) {

	// --- Ftol              Functional change tolerance
	// --- itmax             Maximum number of allowed iterations	
	// --- itmaxlinmin       Maximum number of linmin iterations
	// --- N				 Number of unknowns

	// --- Current and old unknowns vectors, current, old and difference gradients and search direction allocations
	float *d_x;				gpuErrchk(cudaMalloc((void**)&d_x,			N * sizeof(float)));	thrust::device_ptr<float> dev_ptr_x			= thrust::device_pointer_cast(d_x);	
	float *d_Old_x;			gpuErrchk(cudaMalloc((void**)&d_Old_x,		N * sizeof(float)));	
	float *d_Grad;			gpuErrchk(cudaMalloc((void**)&d_Grad,		N * sizeof(float)));	thrust::device_ptr<float> dev_ptr_Grad		= thrust::device_pointer_cast(d_Grad);	
	float *d_Old_Grad;		gpuErrchk(cudaMalloc((void**)&d_Old_Grad,	N * sizeof(float)));	thrust::device_ptr<float> dev_ptr_Old_Grad	= thrust::device_pointer_cast(d_Old_Grad);	
	float *d_diff_Grad;		gpuErrchk(cudaMalloc((void**)&d_diff_Grad,	N * sizeof(float)));	thrust::device_ptr<float> dev_ptr_diff_Grad	= thrust::device_pointer_cast(d_diff_Grad);	
	float *d_XI;			gpuErrchk(cudaMalloc((void**)&d_XI,			N * sizeof(float)));	thrust::device_ptr<float> dev_ptr_XI		= thrust::device_pointer_cast(d_XI);	
	
	// --- Host -> Device memory transfer of the starting point
	gpuErrchk(cudaMemcpy(d_x, h_xstart, N * sizeof(float), cudaMemcpyHostToDevice));
	
	// --- Calculate starting function value and gradient
	float Func = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_x, dev_ptr_x + 1)), thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_x + N - 1, dev_ptr_x + N)), CostFunctionStructGPU(), 0.0f, thrust::plus<float>());
	GradientCalculation<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_x, d_Grad, N);

	printf("Iteration = %i; Functional value = %f\n", 0, 10.f*log10(Func));
	
	// --- Initialize the value of the functional at the previous steps
	float Fold = FLT_MAX;

	// --- Initialize the search direction (XI = -Grad)
	InitXI<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_XI, d_Grad, N);

	// --- Initialize the iterations count
	int count = 1;                              

	// --- Initialize the exit flag
	bool exitflag = 0;                   

	/**************/
	/* ITERATIONS */
	/**************/

	float *h_x = (float *)malloc(N * sizeof(float));
	
	while ((count <= itmax) && (exitflag==0)) {
    
		// --- Save the current value of the unknowns (d_Old_x = d_x)
		gpuErrchk(cudaMemcpy(d_Old_x, d_x, N * sizeof(float), cudaMemcpyDeviceToDevice));

		//  --- Line minimization
		linmin(dev_ptr_x, dev_ptr_XI, N);
   
		// --- Save the current value of the functional
		Fold = Func;

		// --- Save the current value of the gradient (Old_Grad = Grad)
		gpuErrchk(cudaMemcpy(d_Old_Grad, d_Grad, N * sizeof(float), cudaMemcpyDeviceToDevice));

		// --- Update functional and gradient
		Func = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_x, dev_ptr_x + 1)), thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_x + N - 1, dev_ptr_x + N)), CostFunctionStructGPU(), 0.0f, thrust::plus<float>());
		//for (int i=0; i<N; i++) printf("x[%i] = %f\n", i, (float)*(dev_ptr_x+i));
		gpuErrchk(cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));
		//for (int i=0; i<N; i++) printf("x[%i] = %f\n", i, h_x[i]);
		GradientCalculation<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_x, d_Grad, N);
		//for (int i=0; i<N; i++) printf("Grad[%i] = %f\n", i, (float)*(dev_ptr_Grad+i));

		printf("Iteration = %i; Functional value = %f\n", count, 10.f*log10(Func));

		// --- If the new functional value is larger than that at the previous step, then exit
		if (Func > Fold) {
			gpuErrchk(cudaMemcpy(d_x, d_Old_x, N * sizeof(float), cudaMemcpyDeviceToDevice));
			printf("Func > Fold\n");
			exitflag = 1;
		}
    
		// --- If the functional has a small change, then exit
		if (abs(10.f * log10(fabs(Func)) - 10.f * log10(fabs(Fold))) <= Ftol) {
			exitflag = 1;
			printf("Small functional change\n");
		}
    
		if (exitflag == 0) {

			thrust::transform(dev_ptr_Grad, dev_ptr_Grad + N, dev_ptr_Old_Grad, dev_ptr_diff_Grad, thrust::minus<float>());
			float gamma_PR = thrust::inner_product(dev_ptr_diff_Grad,		dev_ptr_diff_Grad + N,		dev_ptr_Grad,		0.0f) / 
							 thrust::inner_product(dev_ptr_Old_Grad,		dev_ptr_Old_Grad + N,		dev_ptr_Old_Grad,		0.0f);
			//thrust::transform(dev_ptr_XI, dev_ptr_XI + N, dev_ptr_Grad, dev_ptr_XI, gamma_PR * _1 - _2);
			thrust::transform(dev_ptr_XI, dev_ptr_XI + N, dev_ptr_Grad, dev_ptr_XI, PR_saxpy_functor(gamma_PR));

			count = count + 1;

		}   
   
	}

	gpuErrchk(cudaMemcpy(h_xfinal, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));
}

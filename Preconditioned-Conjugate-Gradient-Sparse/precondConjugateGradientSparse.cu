include<stdio.h>
#include <math.h>

#include "cublas_v2.h"
#include "cusparse_v2.h"

#include <thrust\device_vector.h>
#include <thrust\functional.h>
#include <thrust\transform.h>
using namespace thrust::placeholders;

#include "Utilities.cuh"
#include "cublasWrappers.cuh"
#include "cusparseWrappers.cuh"

#define BLOCKSIZE 256

// --- Solver parameters - relative tolerance and maximum iterations
#define epsilon 1e-7
#define IMAX 40000

/*******************************************/
/* COMPUTE PRECONDITIONING DIAGONAL MATRIX */
/*******************************************/
template<class T>
__global__ void computePreconditioner(const int Ncols, const int * __restrict__ d_rowIndices_CSR, const int * __restrict__ d_colIndices_CSR, 
	                                  const T * __restrict__ d_AValues_CSR, T * __restrict__ d_M_1)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i<Ncols; i += gridDim.x * blockDim.x)
		for (int k = d_rowIndices_CSR[i] - 1; k < d_rowIndices_CSR[i + 1] - 1; k++)
			if (d_colIndices_CSR[k] == i + 1)
				d_M_1[i] = (T)1 / d_AValues_CSR[k];
}

/*********************************************************************************/
/* PRECONDITIONED CONJUGATE GRADIENT FOR SPARSE MATRICES - DOUBLE PRECISION CASE */
/*********************************************************************************/
template<class T>
void precondConjugateGradientSparse(const int * __restrict__ d_rowIndices_CSR, const int Nn, const int * __restrict__ d_colIndices_CSR, 
	                                const T * __restrict__ d_AValues_CSR, const int nnz, T * __restrict__ d_b, const int Ncols, 
									T * __restrict__ d_x, int &iterations)
{
	cublasHandle_t handleBLAS;
	cusparseHandle_t handleSPARSE;

	cusparseMatDescr_t descrA = 0;

	cublasSafeCall(cublasCreate(&handleBLAS));
	cusparseSafeCall(cusparseCreate(&handleSPARSE));

	T alpha_one = (T)1;
	T beta_zero = (T)0;

	// --- Descriptor for system matrix
	cusparseSafeCall(cusparseCreateMatDescr(&descrA)); 
	cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE)); 

	// Scalars on the host
	T s0, snew, alpha;

	// --- Allocate space for computational vectors on the device
	T* d_M_1;	gpuErrchk(cudaMalloc(&d_M_1,	sizeof(T) * Ncols));		// --- Diagonal matrix for preconditioning
	T *d_r;		gpuErrchk(cudaMalloc(&d_r,		sizeof(T) * Ncols));		// --- Residual
	T *d_z;		gpuErrchk(cudaMalloc(&d_z,		sizeof(T) * Ncols));
	T *d_q;		gpuErrchk(cudaMalloc(&d_q,		sizeof(T) * Ncols));
	T *d_s;		gpuErrchk(cudaMalloc(&d_s,		sizeof(T) * Ncols));

	// Create diagonal preconditioning matrix (J = 1 / diag(M)) 
	computePreconditioner << <1, BLOCKSIZE >> >(Ncols, d_rowIndices_CSR, d_colIndices_CSR, d_AValues_CSR, d_M_1);

	// --- Initialise result vector to zero (ATTENZIONE: QUI SI ASSUME CHE X0 = 0, MA NON DOVREBBE ESSERE NECESSARIAMENTE COSì
	cudaMemset(d_x, 0, Ncols * sizeof(T));

	// --- r_0 = b - A * x_0 (r = b since x = 0)
	gpuErrchk(cudaMemcpy(d_r, d_b, sizeof(T) * Ncols, cudaMemcpyDeviceToDevice));

	// --- z_0 = M^{-1} * r_0
	thrust::transform(thrust::device_pointer_cast(d_M_1), thrust::device_pointer_cast(d_M_1) + Ncols,
					  thrust::device_pointer_cast(d_r),
					  thrust::device_pointer_cast(d_z),
					  thrust::multiplies<T>());

	// --- s0 = <r_0, z_0>
	cublasSafeCall(cublasTdot(handleBLAS, Ncols, d_r, 1, d_z, 1, &s0));
	snew = s0;

	// --- Iterations
	iterations = 0;
	while (iterations < IMAX && snew > epsilon*epsilon*s0)
	{
		// --- q_k = A * z_k
		cusparseSafeCall(cusparseTcsrmv(handleSPARSE, CUSPARSE_OPERATION_NON_TRANSPOSE, Nn - 1, Ncols, nnz, &alpha_one, descrA, d_AValues_CSR, d_rowIndices_CSR, d_colIndices_CSR, d_z, &beta_zero, d_q));
		
		// --- alpha_k = <r_k, z_k> / <z_k, A * z_k>
		cublasSafeCall(cublasTdot(handleBLAS, Ncols, d_z, 1, d_q, 1, &alpha));

		alpha = snew / alpha;
		
		// --- x_{k+1} = x_k + alpha_k * z_k
		cublasSafeCall(cublasTaxpy(handleBLAS, Ncols, &alpha, d_z, 1, d_x, 1));

		// --- r_{k+1} = r_k - alpha_k * A * p_k
		T minusAlpha = -alpha;
		cublasSafeCall(cublasTaxpy(handleBLAS, Ncols, &minusAlpha, d_q, 1, d_r, 1)); 		
		
		// --- s_{k+1} = M^{-1} * r_k
		thrust::transform(thrust::device_pointer_cast(d_M_1), thrust::device_pointer_cast(d_M_1) + Ncols,
						  thrust::device_pointer_cast(d_r),
						  thrust::device_pointer_cast(d_s),
						  thrust::multiplies<T>());

		T sold = snew;
		
		// --- snew = <r_{k+1}, s_{k+1}>
		cublasSafeCall(cublasTdot(handleBLAS, Ncols, d_r, 1, d_s, 1, &snew));
		
		// --- beta_k = snew / sold = <r_{k+1}, s_{k+1}> / <r_0, z_0>;
		T beta = snew / sold;
		
		// --- z_{k+1} = s_{k+1} + beta_k * z_k 
		thrust::transform(thrust::device_pointer_cast(d_z), 
						  thrust::device_pointer_cast(d_z) + Ncols,
						  thrust::device_pointer_cast(d_s), 
						  thrust::device_pointer_cast(d_z), 
						  beta * _1 + _2);

		iterations++;
	}

	// --- Clean up
	gpuErrchk(cudaFree(d_r));
	gpuErrchk(cudaFree(d_z));
	gpuErrchk(cudaFree(d_q));
	gpuErrchk(cudaFree(d_M_1));
}

template void precondConjugateGradientSparse<float>(const int * __restrict__, const int, const int * __restrict__,
	const float * __restrict__, const int, float * __restrict__, const int,
	float * __restrict__, int &);

template void precondConjugateGradientSparse<double>(const int * __restrict__, const int, const int * __restrict__,
	const double * __restrict__, const int, double * __restrict__, const int,
	double * __restrict__, int &);


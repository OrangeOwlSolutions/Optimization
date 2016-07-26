#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "InputOutput.h"
#include "precondConjugateGradientSparse.cuh"

#include "Utilities.cuh"

using namespace std;

#define REAL_TYPE double

/********/
/* MAIN */
/********/
int main() {
	
	const int nnz	= 54924;
	const int Nrows = 7900;
	const int Ncols = 7900;

	cout << "CUDA CG Solver test." << endl << endl;

	cout << "Reading in matrix" << endl;

	int			*h_rowIndices_CSR	= (int *)		malloc((Nrows + 1)	* sizeof(int));
	int			*h_colIndices_CSR	= (int *)		malloc(nnz			* sizeof(int));
	REAL_TYPE	*h_AValues_CSR		= (REAL_TYPE *)	malloc(nnz			* sizeof(REAL_TYPE));

	loadCPUrealtxt(h_AValues_CSR, "D:\\sparsePreconditionedCG\\sparsePreconditionedCG_Approach1\\values.txt", nnz);
	loadCPUrealtxt(h_colIndices_CSR, "D:\\sparsePreconditionedCG\\sparsePreconditionedCG_Approach1\\colIndices.txt", nnz);
	loadCPUrealtxt(h_rowIndices_CSR, "D:\\sparsePreconditionedCG\\sparsePreconditionedCG_Approach1\\rowIndices.txt", Nrows + 1);

	// --- Corrects infinites for single precision case
	for (int k = 0; k < nnz; k++) if (isinf(h_AValues_CSR[k])) h_AValues_CSR[k] = FLT_MAX;

	cout << "Reading in RHS vector" << endl;
	REAL_TYPE *h_b = (REAL_TYPE *)malloc(Nrows * sizeof(REAL_TYPE));
	loadCPUrealtxt(h_b, "D:\\sparsePreconditionedCG\\sparsePreconditionedCG_Approach1\\b.txt", Nrows);

	cout << "Reading in reference solution vector" << endl;
	REAL_TYPE *x = (REAL_TYPE *)malloc(Ncols * sizeof(REAL_TYPE));
	loadCPUrealtxt(x, "D:\\sparsePreconditionedCG\\sparsePreconditionedCG_Approach1\\x.txt", Ncols);

	cout << "Calling solver" << endl;
	REAL_TYPE *h_res = (REAL_TYPE *)malloc(Nrows * sizeof(REAL_TYPE));

	// --- Allocate space for the CSR matrix and host -> device memory copy
	REAL_TYPE *d_AValues_CSR;			gpuErrchk(cudaMalloc((void **)&d_AValues_CSR, sizeof(REAL_TYPE) * nnz));
	gpuErrchk(cudaMemcpy(d_AValues_CSR, h_AValues_CSR, sizeof(REAL_TYPE) * nnz, cudaMemcpyHostToDevice));

	int *d_rowIndices_CSR;			gpuErrchk(cudaMalloc((void **)&d_rowIndices_CSR, sizeof(int) * (Nrows + 1)));
	gpuErrchk(cudaMemcpy(d_rowIndices_CSR, h_rowIndices_CSR, sizeof(int) * (Nrows + 1), cudaMemcpyHostToDevice));

	int *d_colIndices_CSR;			gpuErrchk(cudaMalloc((void **)&d_colIndices_CSR, sizeof(int) * nnz));
	gpuErrchk(cudaMemcpy(d_colIndices_CSR, h_colIndices_CSR, sizeof(int) * nnz, cudaMemcpyHostToDevice));

	// --- Moves the rhs from host to device
	REAL_TYPE *d_b;	cudaMalloc((void **)&d_b, sizeof(REAL_TYPE) * Ncols);
	gpuErrchk(cudaMemcpy(d_b, h_b, sizeof(REAL_TYPE) * Ncols, cudaMemcpyHostToDevice));

	// --- Allocate space for result vector on the device
	REAL_TYPE *d_x;				 gpuErrchk(cudaMalloc((void **)&d_x, sizeof(REAL_TYPE) * Ncols));

	int iterations;
	precondConjugateGradientSparse(d_rowIndices_CSR, Nrows + 1, d_colIndices_CSR, d_AValues_CSR, nnz, d_b, Nrows, d_x, 1, iterations);
	printf("Iterations: %d \n", iterations);

	// --- Copy result back to the host
	gpuErrchk(cudaMemcpy(h_res, d_x, sizeof(REAL_TYPE) * Ncols, cudaMemcpyDeviceToHost));
	
	REAL_TYPE l2norm = h_l2_norm(h_res, x, Ncols);
	cout << "L2 Norm is " << l2norm << endl;

}

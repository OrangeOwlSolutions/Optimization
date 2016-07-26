#include <stdio.h>
#include <float.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GradientDescentCPU.h"
#include "GradientDescentGPU.cuh"

#include "Utilities.cuh"

/********/
/* MAIN */
/********/

int main()
{
	/********************/
	/* INPUT PARAMETERS */
	/********************/
	
	// --- Number of unknowns
	const int M = 5;
	
	// --- Starting point
	float *h_x0 = (float*)malloc(M * sizeof(float));
	for (int i=0; i<M; i++) h_x0[i] = 1.2f;

	// --- Termination tolerance
	const float tol = 1.e-6;

	// --- Maximum number of allowed iterations
	const int maxiter = 10000;

	// --- Step size
	const float alpha = 0.001f;

	// --- Derivative step
	const float h = 0.0001f;

	// --- Minimum allowed perturbations
	const float dxmin = 1e-5;
	
	/*********************/
	/* OUTPUT PARAMETERS */
	/*********************/

	// --- Optimal point
	float* h_xopt = (float*)malloc(M * sizeof(float));
	for (int i=0; i<M; i++) h_xopt[i] = 0.f;

	// --- Optimal functional
	float fopt = 0.f;

	// --- Number of performed iterations
	int niter = 0;

	// --- Gradient norm at optimal point
	float gnorm = 0.f;

	// --- Distance between last and penultimate solutions found
	float dx = 0.f;

	/***************************/
	/* OPTIMIZATION - CPU CASE */
	/***************************/

	GradientDescentCPU(h_x0, tol, maxiter, alpha, h, dxmin, M, h_xopt, &fopt, &niter, &gnorm, &dx);

	printf("Solution found - CPU case:\n");
	printf("fopt = %f; niter = %i; gnorm = %f; dx = %f\n", fopt, niter, gnorm, dx);
	printf("\n\n");
	
#ifdef VERBOSE
	printf("Found minimum - CPU case:\n");
	for (int i=0; i<M; i++) printf("i = %i; h_xopt = %f\n", i, h_xopt[i]);
	printf("\n\n");
#endif
	
	/***************************/
	/* OPTIMIZATION - GPU CASE */
	/***************************/

	// --- Starting point
	float *d_x0;	gpuErrchk(cudaMalloc((void**)&d_x0,		M * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_x0, h_x0, M * sizeof(float), cudaMemcpyHostToDevice));
	// --- Optimal point
	float *d_xopt;	gpuErrchk(cudaMalloc((void**)&d_xopt,	M * sizeof(float)));
	
	GradientDescentGPU(d_x0, tol, maxiter, alpha, h, dxmin, M, d_xopt, &fopt, &niter, &gnorm, &dx);

	printf("Solution found - GPU case:\n");
	printf("fopt = %f; niter = %i; gnorm = %f; dx = %f\n", fopt, niter, gnorm, dx);
	printf("\n\n");

#ifdef VERBOSE
	gpuErrchk(cudaMemcpy(h_xopt, d_xopt, M * sizeof(float), cudaMemcpyDeviceToHost));
	printf("Found minimum - GPU case:\n");
	for (int i=0; i<M; i++) printf("i = %i; h_xopt = %f\n", i, h_xopt[i]);
	printf("\n\n");
#endif
	return 0;
}


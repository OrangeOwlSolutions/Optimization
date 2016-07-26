#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GradientDescentGPU.cuh"

/*******************************/
/* GRADIENT DESCENT - CPU CASE */
/*******************************/
// --- Version using finite differences
//void CostFunctionGradientCPU(float * __restrict h_x, float * __restrict h_g, const float h, const int M) {
//
//	for (int i=0; i<M; i++) {
//		h_x[i] = h_x[i] + h / 2.f;
//		h_g[i] = CostFunction(h_x, M);
//		h_x[i] = h_x[i] - h;
//		h_g[i] = (h_g[i] - CostFunction(h_x, M)) / (2.f * h);
//		h_x[i] = h_x[i] + h / 2.f;
//	}
//}

// --- Version using analytical gradient (Rosenbrock function)
void CostFunctionGradientCPU(float * __restrict h_x, float * __restrict h_g, const float h, const int M) {

	h_g[0] = -400.f * (h_x[1] - h_x[0] * h_x[0]) * h_x[0] + 2.f * (h_x[0] - 1.f);
	for (int i=1; i<M-1; i++) {
		h_g[i]	= -400.f * h_x[i] * (h_x[i+1] - h_x[i] * h_x[i]) + 2.f * (h_x[i] - 1.f) + 200.f * (h_x[i] - h_x[i-1] * h_x[i-1]);
	}
	h_g[M-1] = 200.f * (h_x[M-1] - h_x[M-2] * h_x[M-2]);
}

/********/
/* NORM */
/********/

float normCPU(const float * __restrict h_x, const int M) {

	float sum = 0.f;
	for(int i=0; i<M; i++) sum = sum + h_x[i] * h_x[i];

	return sqrt(sum);

}

/****************************************/
/* GRADIENT DESCENT FUNCTION - CPU CASE */
/****************************************/

// x0      - Starting point
// tol     - Termination tolerance
// maxiter - Maximum number of allowed iterations
// alpha   - Step size
// dxmin   - Minimum allowed perturbations

void GradientDescentCPU(const float * __restrict h_x0, const float tol, const int maxiter, const float alpha, const float h, const float dxmin, const int M,
	                       float * __restrict h_xopt, float *fopt, int *niter, float *gnorm, float *dx) {

	// --- Initialize gradient norm, optimization vector, iteration counter, perturbation

	*gnorm = FLT_MAX; 
	
	float *h_x = (float *)malloc(M * sizeof(float));
	for (int i=0; i<M; i++) h_x[i] = h_x0[i];

	*niter = 0;

	*dx = FLT_MAX;
		
	// --- Allocating space for the gradient, for the new actual solution and for the difference between actual and old solutions
	float *h_g		= (float *)malloc(M * sizeof(float));
	float *h_xnew	= (float *)malloc(M * sizeof(float));
	float *h_xdiff	= (float *)malloc(M * sizeof(float));
	
	// --- Gradient Descent iterations
	while ((*gnorm >= tol) && (*niter <= maxiter) && (*dx >= dxmin)) {
		
	    // --- Calculate gradient
		CostFunctionGradientCPU(h_x, h_g, h, M);
		*gnorm = normCPU(h_g, M);

	    // --- Take step:
		for (int i=0; i<M; i++) h_xnew[i] = h_x[i] - alpha * h_g[i];

		// --- Update termination metrics
		*niter = *niter + 1;
		for (int i=0; i<M; i++) h_xdiff[i] = h_xnew[i] - h_x[i];
		*dx = normCPU(h_xdiff, M);
		for (int i=0; i<M; i++)	h_x[i] = h_xnew[i];
	}

	for (int i=0; i<M; i++) h_xopt[i] = h_x[i];
	*fopt = CostFunction(h_xopt, M);
	*niter = *niter - 1;

}


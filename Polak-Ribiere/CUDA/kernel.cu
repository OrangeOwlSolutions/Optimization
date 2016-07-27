#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "PolakRibiere.cuh"

/********/
/* MAIN */
/********/
int main() {

	const int	N           = 4;              // --- Number of unknowns
	const int	itmax       = 10000;          // --- Maximum number of iterations
	const int	itmaxlinmin = 10;             // --- Maximum number of linmin iterations
	const float Ftol		= 1.e-4;          // --- Functional change tolerance (in dB)

	// --- Memory allocation and initialization of the starting point
	float *h_xstart = (float *)malloc(N * sizeof(float));
	for (int i=0; i<N; i++) h_xstart[i] = 2.f;

	// --- Memory allocation for the final point
	float *h_xfinal = (float *)malloc(N * sizeof(float));

	// --- Polak-Ribière optimization
	PolakRibiere(h_xstart, h_xfinal, itmax, itmaxlinmin, Ftol, N);

	for (int i=0; i<N; i++) printf("xfinal[%i] = %f\n", i, h_xfinal[i]);

}


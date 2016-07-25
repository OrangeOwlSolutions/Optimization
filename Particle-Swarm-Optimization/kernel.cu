// --- Update formulas for EACH particle

// --- v(t) = w * v(t-1) + C1 * R1 * [xbest(t-1) - x(t-1)] + C2 * R2 * [xgbest(t-1) - x(t-1)]

// --- v		: velocity vector
// --- C1, C2	: two positive constants
// --- R1, R2	: random numbers uniformly drawn between 0 and 1
// --- w		: ‘inertia weight’
// --- x		: position vector
// --- xbest	: best-fitness position reached by the particle
// --- xgbest	: is the best-fitness position ever found by the whole swarm

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>

#include <cuda_runtime.h>

#include "Utilities.cuh"
#include "PSO.cuh"

/********/
/* MAIN */
/********/
int main()
{
	cudaSetDevice(0);

	h_PSO_Initialize();
	h_PSO_Optimize();

	return 0;
}


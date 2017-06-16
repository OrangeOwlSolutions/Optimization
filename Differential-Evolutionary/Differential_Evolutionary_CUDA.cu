#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <curand.h>
#include <curand_kernel.h>

using namespace thrust;

#include <stdio.h>
#include <time.h>
#include <fstream>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

#define pi 3.14159265358979f

#define BLOCK_SIZE_POP		32
#define BLOCK_SIZE_RAND1	64
#define BLOCK_SIZE_RAND2	64
#define BLOCK_SIZE_UNKN		8
#define BLOCK_SIZE			256

#define PI_f				3.14159265358979f

#define TIMING
//#define SHARED_VERSION
//#define REGISTER_VERSION

//#define DEBUG

//#define ANTENNAS

// --- REFERENCES
//     [1] R. Storn and K. Price, “Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces,” 
//     Journal of Global Optimization, vol. 11, no. 4, pp. 341–359, 1997

//     [2] Lucas de P. Veronese and Renato A. Krohling, “Differential Evolution Algorithm on the GPU with C-CUDA,” 
//     Proc. of the IEEE Congress on Evolutionary Computation, Barcelona, Spain, Jul. 18-23, 2010, pp. 1-7.

// Conventions: the index j addresses the population member while the index i addresses the member component
//              the homologous host and device variables have the same name with a "h_" or "d_" prefix, respectively
//				the __host__ and __device__ functions pointer parameters have the same name for comparison purposes. it is up to the caller to use 
//				host or device pointers, as appropriate

/****************************************/
/* EVALUATION OF THE OBJECTIVE FUNCTION */
/****************************************/
__global__ void curand_setup_kernel(curandState * __restrict state, const unsigned long int seed)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	curand_init(seed, tid, 0, &state[tid]);
}

/********************************/
/* INITIALIZE POPULATION ON GPU */
/********************************/
__global__ void initialize_population_GPU(float * __restrict pop, const float * __restrict minima, const float * __restrict maxima,
	curandState * __restrict state, const int D, const int Np) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < Np)) pop[j*D + i] = (maxima[i] - minima[i]) * curand_uniform(&state[j*D + i]) + minima[i];
}

/****************************************/
/* EVALUATION OF THE OBJECTIVE FUNCTION */
/****************************************/
#ifndef ANTENNAS
__host__ __device__ float functional(const float * __restrict x, const int D) {

	// --- More functionals at https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/ortizboyer05a-html/node6.html
	
	float sum;
	// --- De Jong function (hypersphere)
	//#define MINIMA -5.12
	//#define MAXIMA  5.12
	//sum = 0.f;
	//for (int i=0; i<D; i++) sum = sum + x[i] * x[i];
	// --- Rosenbrock's saddle - xopt = (1., 1., ..., 1.)
	#define MINIMA -2.048
	#define MAXIMA  2.048
	sum = 0.f;
	for (int i = 1; i<D; i++) sum = sum + 100.f * (x[i] - x[i - 1] * x[i - 1]) * (x[i] - x[i - 1] * x[i - 1]) + (x[i - 1] - 1.f) * (x[i - 1] - 1.f);
	// --- Rastrigin - xopt = (0., 0., ..., 0.)
	//#define MINIMA -5.12
	//#define MAXIMA  5.12
	//sum = 10.f * D;
	//for (int i = 1; i <= D; i++) sum = sum + (x[i - 1] * x[i - 1] - 10.f * cos(2.f * PI_f * x[i - 1]));
	// --- Schwfel - xopt(-420.9698, -420.9698, ..., -420.9698)
	//#define MINIMA -512.03
	//#define MAXIMA  511.97
	//sum = 418.9829 * D;
	//for (int i = 1; i <= D; i++) sum = sum + x[i - 1] * sin(sqrt(fabs(x[i - 1])));

	return sum;
}
#else
#define MINIMA -PI_f
#define MAXIMA  PI_f
__host__ __device__ float cheb(const float x, const int N) {

	if (fabs(x) <= 1.f) return cos(N * acos(x));
	else				return cosh(N * acosh(x));

}

__host__ __device__ float pattern(const float u, const float beta, const int N) {

	const float temp = cheb(u / (beta * 0.3), N);

	return 1.f / sqrt(1.f + 0.1 * fabs(temp) * fabs(temp));

}

__host__ __device__ float functional(float* x, int D, int N, float d, float beta, float Deltau) {

	// --- Functional value
	float sum = 0.f;

	// --- Spectral variable
	float u;

	// --- Real and imaginary parts of the array factor and squared absolute value
	float Fr, Fi, F2, Frref, Firef, F2ref;
	// --- Reference pattern (ASSUMED real for simplicity!)
	float R;
	// --- Maximum absolute value of the array factor
	float maxF = -FLT_MAX;

	// --- Calculating the array factor and the maximum of its absolute value
	for (int i = 0; i<N; i++) {
		u = -beta + i * Deltau;
		Fr = Fi = 0.;
		Frref = Firef = 0.;

		for (int j = 0; j<D; j++) {
			Fr = Fr + cos(j * u * d + x[j]);
			Fi = Fi + sin(j * u * d + x[j]);
		}
		F2 = Fr * Fr + Fi * Fi;
		//F2ref = (3.f * cos(u / (0.5*beta)) * cos(u / (0.5*beta)) * cos(u / (0.5*beta))) * (3.f * cos(u / (0.5*beta)));
		F2ref = (3.f * cos((u - 0.1*beta) / (0.5*beta)) * cos((u - 0.1*beta) / (0.5*beta)) * cos((u - 0.1*beta) / (0.5*beta))) * (3.f * cos((u - 0.1*beta) / (0.5*beta)));
		//F2ref = 2.f * pattern(u, beta, N);
		//F2ref = F2ref * F2ref;
		sum = sum + (F2 - F2ref) * (F2 - F2ref);
	}

	return sum;
}
#endif

/********************************/
/* POPULATION EVALUATION ON GPU */
/********************************/
#ifndef ANTENNAS
__global__ void evaluation_GPU(const int Np, const int D, const float * __restrict pop, float * __restrict fobj) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j < Np)  fobj[j] = functional(&pop[j*D], D);
}
#else
__global__ void evaluation_GPU(int Np, int D, float *pop, float *fobj, int N, float d, float beta, float Deltau) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j < Np) fobj[j] = functional(&pop[j*D], D, N, d, beta, Deltau);

}
#endif

/**********************************************************/
/* GENERATE MUTATION INDICES AND CROSS-OVER VALUES ON GPU */
/**********************************************************/
__global__ void generate_crossover_values_GPU(float * __restrict Rand, const int Np, const int D, curandState * __restrict state) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	//if (j < D * Np) Rand[j] = curand_uniform(&state[j*Np]);
	if (j < D * Np) Rand[j] = curand_uniform(&state[j]);
}

/**********************************************************/
/* GENERATE MUTATION INDICES AND CROSS-OVER VALUES ON GPU */
/**********************************************************/
__global__ void generate_mutation_indices_GPU(int * __restrict mutation, const int Np, const int D, curandState * __restrict state) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int a, b, c;

	if (j < Np) {

		//do a = Np*(curand_uniform(&state[j*D]));	while (a == j);
		//do b = Np*(curand_uniform(&state[j*D]));	while (b == j || b == a);
		//do c = Np*(curand_uniform(&state[j*D]));	while (c == j || c == a || c == b);
		do a = Np*(curand_uniform(&state[j]));	while (a == j);
		do b = Np*(curand_uniform(&state[j]));	while (b == j || b == a);
		do c = Np*(curand_uniform(&state[j]));	while (c == j || c == a || c == b);
		mutation[j * 3] = a;
		mutation[j * 3 + 1] = b;
		mutation[j * 3 + 2] = c;

	}
}

/**********************************/
/* GENERATION OF A NEW POPULATION */
/**********************************/
__global__ void generation_new_population_GPU(const float * __restrict pop, const int NP, const int D, float * __restrict npop, const float F,
	const float CR, const float * __restrict rand, const int * __restrict mutation,
	const float * __restrict minimum, const float * __restrict maximum) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < NP)) {

		// --- Mutation indices
		int a = mutation[j * 3];
		int b = mutation[j * 3 + 1];
		int c = mutation[j * 3 + 2];

		// --- Mutation and binomial crossover
		// --- DE/rand/1. One of the best strategies. Try F = 0.7 and CR = 0.5 as a first guess.
		//if (rand[j*D + i]<CR)	npop[j*D + i] = pop[a*D + i] + F*(pop[b*D + i] - pop[c*D + i]);
		if (rand[j]<CR)	npop[j*D + i] = pop[a*D + i] + F*(pop[b*D + i] - pop[c*D + i]);
		else			npop[j*D + i] = pop[j*D + i];
		//printf("%f\n", npop[j*D + i]);

		// --- Other possible approaches to mutation and crossover
		// --- DE/best/1 --- Not bad, but found several optimization problems where misconvergence occurs.
		//npop[j*D+i] = pop[best_old_gen_ind*D+i] + F*(pop[b*D+i]-pop[c*D+i]);
		// --- DE/rand to best/1 --- F1 can be different or equal to F2
		//npop[j*D+i] = pop[a*D+i] + F1*(pop[best_old_gen_ind*D+i] - pop[a*D+i]) + F2*(pop[b*D+i]-pop[c*D+i]);
		// --- DE/current to best/1 --- One of the best strategies. Try F = 0.85 and CR = 1. In case of misconvergence, try to increase NP. If this doesn't help,
		//     play around with all the control variables --- F1 can be different or equal to F2
		//npop[j*D+i] = pop[j*D+i] + F1*(pop[best_old_gen_ind*D+i] - pop[j*D+i]) + F2*(pop[a*D+i]-pop[b*D+i]);
		// --- DE/rand/2 --- Robust optimizer for many functions.
		//npop[j*D+i] = pop[e*D+i] + F*(pop[a*D+i]+pop[b*D+i]-pop[c*D+i]-pop[d*D+i]);
		// --- DE/best/2 --- Powerful strategy worth trying.
		//npop[j*D+i] = pop[best_old_gen_ind*D+i] + F*(pop[a*D+i]+pop[b*D+i]-pop[c*D+i]-pop[d*D+i]);

		// --- Saturation due to constraints on the unknown parameters
		if (npop[j*D + i]>maximum[i])		npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])	npop[j*D + i] = minimum[i];

	}

}

/*******************************/
/* POPULATION SELECTION ON GPU */
/*******************************/
// Assumption: all the optimization variables are associated to the same thread block
#ifndef ANTENNAS
__global__ void selection_and_evaluation_GPU(const int Np, const int D, float * __restrict pop, const float * __restrict npop, float * __restrict fobj) {

	int i = threadIdx.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < Np)) {

		float nfobj = functional(&npop[j*D], D);

		float temp = fobj[j];

		if (nfobj < temp) {
			pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj;
		}
	}
}
#else
// Assumption: all the optimization variables are associated to the same thread block
__global__ void selection_and_evaluation_GPU(int Np, int D, float *pop, float *npop, float *fobj, int N, float d, float beta, float Deltau) {

	int i = threadIdx.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < Np)) {

		float nfobj = functional(&npop[j*D], D, N, d, beta, Deltau);

		float temp = fobj[j];

		if (nfobj < temp) {
			pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj;
		}
	}
}
#endif

/***********************************************************************************/
/* GENERATION OF A NEW POPULATION, MUTATION, CROSSOVER AND SELECTION - GPU VERSION */
/***********************************************************************************/
#ifdef SHARED_VERSION
// --- It assumes that BLOCK_SIZE_POP >= D
__global__ void generation_new_population_mutation_crossover_selection_evaluation_GPU(float * __restrict__ pop, const int Np, const int D, float * __restrict__ npop, const float F, const float CR,
																					  const float * __restrict__ minimum, float * __restrict__ maximum, float * __restrict__ fobj, curandState * __restrict__ state) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// --- Shared memory is used as a controlled cache
	__shared__ int		a[BLOCK_SIZE_POP], b[BLOCK_SIZE_POP], c[BLOCK_SIZE_POP];
	//__shared__ float	Rand[BLOCK_SIZE_POP], nfobj[BLOCK_SIZE_POP], temp[BLOCK_SIZE_POP];
	__shared__ float	nfobj[BLOCK_SIZE_POP], temp[BLOCK_SIZE_POP];

	// --- Generate mutation indices and crossover values
	if ((i == 0) && (j < Np)) {

		// --- Mutation indices
		do a[threadIdx.y] = Np*(curand_uniform(&state[j]));	while (a[threadIdx.y] == j);
		do b[threadIdx.y] = Np*(curand_uniform(&state[j]));	while (b[threadIdx.y] == j || b[threadIdx.y] == a[threadIdx.y]);
		do c[threadIdx.y] = Np*(curand_uniform(&state[j]));	while (c[threadIdx.y] == j || c[threadIdx.y] == a[threadIdx.y] || b[threadIdx.y] == a[threadIdx.y]);

		//// --- Crossover values
		//Rand[threadIdx.y] = curand_uniform(&state[j]);
	}

	__syncthreads();

	// --- Generate new population
	if ((i < D) && (j < Np)) {

		// --- Crossover values
		float Rand = curand_uniform(&state[j]);
		
		// --- Mutation and crossover
		//if (Rand[threadIdx.y] < CR)	npop[j*D + i] = pop[a[threadIdx.y] * D + i] + F*(pop[b[threadIdx.y] * D + i] - pop[c[threadIdx.y] * D + i]);
		if (Rand < CR)	npop[j*D + i] = pop[a[threadIdx.y] * D + i] + F*(pop[b[threadIdx.y] * D + i] - pop[c[threadIdx.y] * D + i]);
		else			npop[j*D + i] = pop[j*D + i];

		// --- Saturation due to constraints on the unknown parameters
		if (npop[j*D + i]>maximum[i]) npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])npop[j*D + i] = minimum[i];

	}

	__threadfence();

	if ((i == 0) && (j < Np)) {

		// --- Evaluation and selection
		nfobj[threadIdx.y] = functional(&npop[j*D], D);

		temp[threadIdx.y] = fobj[j];

	}

	__syncthreads();

	if ((i < D) && (j < Np)) {
		if (nfobj[threadIdx.y] < temp[threadIdx.y]) {
			pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj[threadIdx.y];
		}

	}
}
#endif

#ifdef REGISTER_VERSION
__global__ void generation_new_population_mutation_crossover_selection_evaluation_GPU(float * __restrict__ pop, const int Np, const int D,
	float * __restrict__ npop, const float F, const float CR,
	const float * __restrict__ minimum, float * __restrict__ maximum,
	float * __restrict__ fobj, curandState * __restrict__ state) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	// --- Generate mutation indices and crossover values

	int a, b, c;
	float Rand;

	if (j < Np) {

		// --- Mutation indices
		do a = Np*(curand_uniform(&state[j]));	while (a == j);
		do b = Np*(curand_uniform(&state[j]));	while (b == j || b == a);
		do c = Np*(curand_uniform(&state[j]));	while (c == j || c == a || b == a);

		//// --- Crossover values
		//Rand = curand_uniform(&state[j]);

		// --- Generate new population

		// --- Mutation and crossover
		//if (Rand < CR)	for (int i = 0; i<D; i++) npop[j*D + i] = pop[a*D + i] + F*(pop[b*D + i] - pop[c*D + i]);
		//else			for (int i = 0; i<D; i++) npop[j*D + i] = pop[j*D + i];
		for (int i = 0; i<D; i++) {
			// --- Crossover values
			Rand = curand_uniform(&state[j]);
			if (Rand < CR) npop[j*D + i] = pop[a*D + i] + F*(pop[b*D + i] - pop[c*D + i]);
			else           npop[j*D + i] = pop[j*D + i];
		}

		// --- Saturation due to constraints on the unknown parameters
		for (int i = 0; i<D; i++) if (npop[j*D + i]>maximum[i]) npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])npop[j*D + i] = minimum[i];

		// --- Evaluation and selection
		float nfobj = functional(&npop[j*D], D);

		float temp = fobj[j];

		if (nfobj < temp) {
			for (int i = 0; i<D; i++) pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj;
		}

	}
}
#endif

/***********************/
/* FIND MINIMUM ON GPU */
/***********************/
void find_minimum_GPU(const int N, float *t, float * __restrict minval, int * __restrict index) {

	// --- Wrap raw pointer with a device_ptr 
	device_ptr<float> dev_ptr = device_pointer_cast(t);

	// --- Use device_ptr in thrust min_element
	device_ptr<float> min_ptr = thrust::min_element(dev_ptr, dev_ptr + N);

	index[0] = &min_ptr[0] - &dev_ptr[0];

	minval[0] = min_ptr[0];;

}

/********/
/* MAIN */
/********/
int main()
{
	// --- Number of individuals in the population (Np >=4 for mutation purposes)
	int			Np = 400;
	// --- Dimensionality of each individual (number of unknowns)
	int			D = 5;
	// --- Mutation factor (0 < F <= 2). Typically chosen in [0.5, 1], see Ref. [1]
	float		F = 0.7f;
	// --- Maximum number of generations
	int			Gmax = 8000;
	// --- Crossover constant (0 < CR <= 1)
	float		CR = 0.4f;

	// --- Mutually different random integer indices selected from {1, 2, … ,Np}
	int *d_mutation,			// --- Device side mutation vector
		*h_best_index_dev;		// --- Host side current optimal member index of device side
	//int *d_best_index;			// --- Device side current optimal member index

#ifdef ANTENNAS
	// --- Wavelength
	float		lambda = 1.f;
	// --- Interelement distance
	float		d = lambda / 2.f;
	// --- Wavenumber
	float		beta = 2.f*pi / lambda;
	// --- Spectral oversampling factor
	float		overs = 4.f;
	// --- Sampling step in the spectral domain
	float		Deltau = pi / (overs*(D - 1)*d);
	// --- Number of spectral sampling points
	int			N = floor(4 * (D - 1)*d*overs / lambda);
#endif

	float *d_pop,				// --- Device side population
		*d_npop,					// --- Device side new population (trial vectors)
		*d_Rand,					// --- Device side crossover rand vector (uniformly distributed in (0,1))
		*d_fobj,					// --- Device side objective function value
		*d_maxima,					// --- Device side maximum constraints vector
		*d_minima,					// --- Device side minimum constraints vector
		*h_pop_dev_res,				// --- Host side population result of GPU computations
		*h_best_dev,				// --- Host side population best value history of device side
		*h_maxima,					// --- Host side maximum constraints vector
		*h_minima;					// --- Host side minimum constraints vector

	curandState *devState;		// --- Device side random generator state vector

	// --- Device side memory allocations
	gpuErrchk(cudaMalloc((void**)&d_pop, D*Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_npop, D*Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_Rand, D*Np*sizeof(float)));
	//gpuErrchk(cudaMalloc((void**)&d_Rand, Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj, Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_mutation, 3 * Np*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_maxima, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&devState, D*Np*sizeof(curandState)));

	// --- Host side memory allocations
	h_pop_dev_res = (float*)malloc(D*Np*sizeof(float));
	h_best_dev = (float*)malloc(Gmax*sizeof(float));
	h_best_index_dev = (int*)malloc(Gmax*sizeof(int));
	h_maxima = (float*)malloc(D*sizeof(float));
	h_minima = (float*)malloc(D*sizeof(float));

	// --- Define grid sizes
	int Num_Blocks_Pop = iDivUp(Np, BLOCK_SIZE_POP);
	dim3 Grid(iDivUp(D, BLOCK_SIZE_UNKN), iDivUp(Np, BLOCK_SIZE_POP));
	dim3 Block(BLOCK_SIZE_UNKN, BLOCK_SIZE_POP);

	// --- Set maxima and minima
	for (int i = 0; i<D; i++) {
		h_maxima[i] = MAXIMA;
		h_minima[i] = MINIMA;
	}
	gpuErrchk(cudaMemcpy(d_maxima, h_maxima, D*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_minima, h_minima, D*sizeof(float), cudaMemcpyHostToDevice));

	// --- Initialize cuRAND states
	curand_setup_kernel << <iDivUp(D*Np, BLOCK_SIZE), BLOCK_SIZE >> >(devState, time(NULL));

	// --- Initialize popultion
	initialize_population_GPU << <Grid, Block >> >(d_pop, d_minima, d_maxima, devState, D, Np);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// --- Evaluate population
#ifndef ANTENNAS
	evaluation_GPU << <iDivUp(Np, BLOCK_SIZE), BLOCK_SIZE >> >(Np, D, d_pop, d_fobj);
#else
	evaluation_GPU<<<Num_Blocks_Pop,BLOCK_SIZE_POP>>>(Np, D, d_pop, d_fobj, N, d, beta, Deltau);
#endif
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	TimingGPU timerGPU;
	timerGPU.StartCounter();
	for (int i = 0; i<Gmax; i++) {

#ifdef SHARED_VERSION
		generation_new_population_mutation_crossover_selection_evaluation_GPU << <Grid, Block >> >(d_pop, Np, D, d_npop, F, CR, d_minima, d_maxima, d_fobj, devState);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
#elif defined REGISTER_VERSION
		generation_new_population_mutation_crossover_selection_evaluation_GPU<<<iDivUp(Np,BLOCK_SIZE_POP), BLOCK_SIZE_POP>>>(d_pop, Np, D, d_npop, F, CR, d_minima, d_maxima, d_fobj, devState);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
#else
		// --- Generate mutation indices 
		generate_mutation_indices_GPU << <iDivUp(Np, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation, Np, D, devState);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Generate crossover values 
		generate_crossover_values_GPU << <iDivUp(D * Np, BLOCK_SIZE_RAND2), BLOCK_SIZE_RAND2 >> >(d_Rand, Np, D, devState);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Generate new population
		generation_new_population_GPU << <Grid, Block >> >(d_pop, Np, D, d_npop, F, CR, d_Rand, d_mutation, d_minima, d_maxima);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Select new population and evaluate it
#ifndef ANTENNAS
		selection_and_evaluation_GPU << <Grid, Block >> >(Np, D, d_pop, d_npop, d_fobj);
#else
		selection_and_evaluation_GPU << <Grid, Block >> >(Np, D, d_pop, d_npop, d_fobj, N, d, beta, Deltau);
#endif
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
#endif
		find_minimum_GPU(Np, d_fobj, &h_best_dev[i], &h_best_index_dev[i]);

#ifndef TIMING
		printf("Iteration: %i; best member value: %f: best member index: %i\n", i, h_best_dev[i], h_best_index_dev[i]);
#endif

	}
#ifdef TIMING
	printf("Total timing = %f [ms]\n", timerGPU.GetCounter());
#endif TIMING

	gpuErrchk(cudaMemcpy(h_pop_dev_res, d_pop, Np*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i<D; i++) printf("Variable nr. %i = %f\n", i, h_pop_dev_res[h_best_index_dev[Gmax - 1] * D + i]);

	return 0;
}

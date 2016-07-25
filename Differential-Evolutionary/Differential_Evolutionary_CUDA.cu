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

#define pi 3.14159265358979f

#define BLOCK_SIZE_POP	32
#define BLOCK_SIZE_RAND	64
#define BLOCK_SIZE_UNKN 8
#define BLOCK_SIZE		256

//#define DEBUG

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
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed, tid, 0, &state[tid]);
}

/********************************/
/* INITIALIZE POPULATION ON GPU */
/********************************/
__global__ void initialize_population_GPU(float * __restrict pop, const float * __restrict minima, const float * __restrict maxima, 
	                                      curandState * __restrict state, const int D, const int Np) {
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if ((i < D) && (j < Np)) pop[j*D+i] = (maxima[i] - minima[i]) * curand_uniform(&state[j*D+i]) + minima[i];
}

/****************************************/
/* EVALUATION OF THE OBJECTIVE FUNCTION */
/****************************************/
__host__ __device__ float functional(const float * __restrict x, const int D) {

	float sum = 0.f;

	// --- De Jong function
	//for (int i=0; i<D; i++) sum = sum + x[i] * x[i];
	// --- Rosenbrock's saddle
	sum = 0.f;
	for (int i=1; i<D; i++) sum = sum + 100.f * (x[i] - x[i-1] * x[i-1]) * (x[i] - x[i-1] * x[i-1]) + (x[i-1] - 1.f) * (x[i-1] - 1.f);

	return sum;
}

/********************************/
/* POPULATION EVALUATION ON GPU */
/********************************/
__global__ void evaluation_GPU(const int Np, const int D, const float * __restrict pop, float * __restrict fobj) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (j < Np)  fobj[j] = functional(&pop[j*D], D);
}

/**********************************************************/
/* GENERATE MUTATION INDICES AND CROSS-OVER VALUES ON GPU */
/**********************************************************/
__global__ void generate_mutation_indices_and_crossover_values_GPU(float * __restrict Rand, int * __restrict mutation, const int Np, const int D,
	                                                               curandState * __restrict state) {
	
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int a, b, c;
	
	if (j < Np) {

		do a=Np*(curand_uniform(&state[j*D]));	while(a==j);
		do b=Np*(curand_uniform(&state[j*D]));	while(b==j||b==a);
		do c=Np*(curand_uniform(&state[j*D]));	while(c==j||c==a||c==b);
		mutation[j*3]=a;
		mutation[j*3+1]=b;
		mutation[j*3+2]=c;

		Rand[j]=curand_uniform(&state[j*D]);
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
		int a=mutation[j*3];
		int b=mutation[j*3+1];
		int c=mutation[j*3+2];

		// --- Mutation and crossover
		// --- One of the best strategies. Try F = 0.7 and CR = 0.5 as a first guess.
		if(rand[j]<CR)	npop[j*D+i] = pop[a*D+i]+F*(pop[b*D+i]-pop[c*D+i]);
		else			npop[j*D+i] = pop[j*D+i];

		// --- Other possible approaches to mutation and crossover
		// --- Not bad, but found several optimization problems where misconvergence occurs.
		//npop[j*D+i] = pop[best_old_gen_ind*D+i] + F*(pop[b*D+i]-pop_old[c*D+i]);
		// --- One of the best strategies. Try F = 0.85 and CR = 1. In case of misconvergence, try to increase NP. If this doesn't help,
		//     play around with all the control variables.
		//npop[j*D+i] = pop[j*D+i] + F*(pop[best_old_gen_ind*D+i] - pop[j*D+i]) + F*(pop[a*D+i]-pop[b*D+i]);
		// --- Powerful strategy worth trying.
		//npop[j*D+i] = pop[best_old_gen_ind*D+i] + (pop[a*D+i]+pop[b*D+i]-pop[c*D+i]-pop[d*D+i])*F;
		// --- Robust optimizer for many functions.
		//npop[j*D+i] = pop[e*D+i] + (pop[a*D+i]+pop[b*D+i]-pop[c*D+i]-pop[d*D+i])*F;

		// --- Saturation due to constraints on the unknown parameters
		if		(npop[j*D+i]>maximum[i])	npop[j*D+i]=maximum[i];
		else if	(npop[j*D+i]<minimum[i])	npop[j*D+i]=minimum[i];

	}
	
}

/*******************************/
/* POPULATION SELECTION ON GPU */
/*******************************/
// Assumption: all the optimization variables are associated to the same thread block
__global__ void selection_and_evaluation_GPU(const int Np, const int D, float * __restrict pop, const float * __restrict npop, float * __restrict fobj) {

	int i = threadIdx.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if ((i < D) && (j < Np)) {

		float nfobj = functional(&npop[j*D], D);

		float temp = fobj[j];

		if (nfobj < temp) { 
			pop[j*D+i]	= npop[j*D+i];
			fobj[j]		= nfobj;
		}
	}
}

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
	int			Np		= 80;  
	// --- Dimensionality of each individual (number of unknowns)
	int			D		= 5;
	// --- Mutation factor (0 < F <= 2). Typically chosen in [0.5, 1], see Ref. [1]
	float		F		= 0.7f;
	// --- Maximum number of generations
	int			Gmax	= 2000;
	// --- Crossover constant (0 < CR <= 1)
	float		CR		= 0.4f;

	// --- Mutually different random integer indices selected from {1, 2, … ,Np}
	int *d_mutation,			// --- Device side mutation vector
		*d_best_index,			// --- Device side current optimal member index
		*h_best_index_dev;		// --- Host side current optimal member index of device side

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
	gpuErrchk(cudaMalloc((void**)&d_pop,D*Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_npop,D*Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_Rand,Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj,Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_mutation,3*Np*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_maxima,D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima,D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&devState, D*Np*sizeof(curandState)));

	// --- Host side memory allocations
	h_pop_dev_res		= (float*)malloc(D*Np*sizeof(float));
	h_best_dev			= (float*)malloc(Gmax*sizeof(float));
	h_best_index_dev	= (int*)malloc(Gmax*sizeof(int));
	h_maxima			= (float*)malloc(D*sizeof(float));
	h_minima			= (float*)malloc(D*sizeof(float));

	// --- Define grid sizes
	int Num_Blocks_Pop		= iDivUp(Np,BLOCK_SIZE_POP);
	int Num_Blocks_Rand2	= iDivUp(Np,BLOCK_SIZE_RAND);
	dim3 Grid(iDivUp(D,BLOCK_SIZE_UNKN),iDivUp(Np,BLOCK_SIZE_POP));
	dim3 Block(BLOCK_SIZE_UNKN,BLOCK_SIZE_POP);

	// --- Set maxima and minima
	for (int i=0; i<D; i++) {
		h_maxima[i] =  2.;
		h_minima[i] = -2.;
	}
	gpuErrchk(cudaMemcpy(d_maxima, h_maxima, D*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_minima, h_minima, D*sizeof(float), cudaMemcpyHostToDevice));
	
	// --- Initialize cuRAND states
	curand_setup_kernel<<<iDivUp(D*Np, BLOCK_SIZE), BLOCK_SIZE>>>(devState, time(NULL));
	
	// --- Initialize popultion
	initialize_population_GPU<<<Grid, Block>>>(d_pop, d_minima, d_maxima, devState, D, Np);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// --- Evaluate population
	evaluation_GPU<<<iDivUp(Np, BLOCK_SIZE), BLOCK_SIZE>>>(Np, D, d_pop, d_fobj);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	int a, b, c;
	for(int i=0;i<Gmax;i++) {
		
		// --- Generate mutation indices and cross-over uniformly distributed random vector
		generate_mutation_indices_and_crossover_values_GPU<<<Num_Blocks_Rand2,BLOCK_SIZE_RAND>>>(d_Rand, d_mutation, Np, D, devState);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Generate new population
		generation_new_population_GPU<<<Grid,Block>>>(d_pop, Np, D, d_npop, F, CR, d_Rand, d_mutation, d_minima, d_maxima);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Select new population and evaluate it
		selection_and_evaluation_GPU<<<Grid,Block>>>(Np, D, d_pop, d_npop, d_fobj);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		find_minimum_GPU(Np, d_fobj, &h_best_dev[i], &h_best_index_dev[i]);
		
		printf("Iteration: %i; best member value: %f: best member index: %i\n", i, h_best_dev[i], h_best_index_dev[i]);

	}

	gpuErrchk(cudaMemcpy(h_pop_dev_res, d_pop, Np*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i=0; i<D; i++) printf("Variable nr. %i = %f\n", i, h_pop_dev_res[h_best_index_dev[Gmax-1]*D+i]);

	return 0;
}


#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <cub/cub.cuh>

#include "PSO.cuh"
#include "Utilities.cuh"
#include "Functional.cuh"

#define BLOCKSIZE_LOCAL_BEST   256

#define DEBUG

texture<float, 1, cudaReadModeElementType> functional_texture;

/******************/
/* PSO PARAMETERS */
/******************/
#define numberOfParticles		10							// --- Number of individuals
#define numberOfUnknowns		 5							// --- Number of unknowns
#define numberOfGenerations    100							// --- Number of iterations
#define RADIUS					 2							// --- Radius of the ring neighboorhood to which each particle communicates
#define W						 0.7213475204444817278f		// --- Inertia weight W
#define C1						 1.193147180559945286f		// --- Cognitive attraction factor C1
#define C2						 1.193147180559945286f		// --- Social attraction factor C2
#define BLOCKSIZE				 (numberOfUnknowns)

/********************/
/* GLOBAL VARIABLES */
/********************/
curandState		*devStates; 
unsigned int	actualParticleSize;
unsigned int	*d_localBestIDs; 
unsigned int	*d_to_be_updated; 
int				vectorSwarmSize;
int				scalarFloatSwarmSize;
int				scalarIntSwarmSize;
float			h_globalBestFitness;					// --- Final global best fitness value
float			*h_globalBestPosition;
float			*d_positions; 
float			*d_best_personal_positions; 
float			*d_velocities; 
float			*d_functionals; 
float			*d_personal_best_functional; 

// --- Lower limit for each dimension of the search space MIN_VALS
const float MINVALS[numberOfUnknowns] = {-5.12f, -5.12f, -5.12f, -5.12f, -5.12f};
// --- Upper limit for each dimension of the search space MAX_VALS
const float MAXVALS[numberOfUnknowns] = { 5.12f,  5.12f,  5.12f,  5.12f,  5.12f};

float DELTAVALS[numberOfUnknowns];

/************************/
/* CONSTANT MEMORY DATA */
/************************/
// --- Starting coordinates of the hypercubical search space 
__constant__ float  c_minValues[numberOfUnknowns];

// --- Ending coordinates of the hypercubical search space 
__constant__ float  c_maxValues[numberOfUnknowns];

// --- Widths of the hypercubical search space 
__constant__ float  c_deltaValues[numberOfUnknowns];

/*********************************************/
/* INITIALIZATION OF RANDOM NUMBER GENERATOR */
/*********************************************/
__global__ void init_pseudorandom_generator(curandState * __restrict__ state, const unsigned long seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, id, 0, &state[id]);
}

/********************************************************/
/* INITIALIZATIONS OF PARTICLE POSITIONS AND VELOCITIES */
/********************************************************/
__global__ void particles_initialization(float * __restrict__ d_positions,  float * __restrict__ d_best_personal_positions, 
	                                     float * __restrict__ d_velocities, curandState * __restrict__ devStates) {

	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// --- Position initializations
	float R = curand_uniform(&devStates[tid]);
	float pos = c_minValues[threadIdx.x] + R * c_deltaValues[threadIdx.x];
	d_positions[tid]     = pos;
	d_best_personal_positions[tid] = pos;

	// --- Velocity initializations
	R = curand_uniform(&devStates[tid]);
	float vel = c_minValues[threadIdx.x] + R * c_deltaValues[threadIdx.x];
	d_velocities[tid] = (vel - pos) / 2.0;
}

/**********************/
/* PSO INITIALIZATION */
/**********************/
void h_PSO_Initialize() {

	// To achieve byte alignment, we need data arrays with a number of elements for each particle which is a multiple of 16
	//unsigned int actualParticleSize = iAlignUp(numberOfUnknowns, 16);
	actualParticleSize = numberOfUnknowns;

	vectorSwarmSize = numberOfParticles * actualParticleSize * sizeof(float);

	// --- Initializing the positions (numberOfParticles * actualParticleSize)
	gpuErrchk(cudaMalloc((void**)&d_positions, vectorSwarmSize));
	gpuErrchk(cudaMemset(d_positions, 0, vectorSwarmSize));

	// --- Allocation of the current personal best positions array (numberOfParticles * actualParticleSize)
	gpuErrchk(cudaMalloc((void**)&d_best_personal_positions, vectorSwarmSize));
	gpuErrchk(cudaMemset(d_best_personal_positions, 0, vectorSwarmSize));

	// --- Initializing the current velocities (numberOfParticles * actualParticleSize)
	gpuErrchk(cudaMalloc((void**)&d_velocities, vectorSwarmSize));
	gpuErrchk(cudaMemset(d_velocities, 0, vectorSwarmSize));

	scalarFloatSwarmSize = numberOfParticles * sizeof(float);

	// --- Initializing fitnesses (numberOfParticles) and binding the texture
	size_t pitch;
	gpuErrchk(cudaMallocPitch(&d_functionals, &pitch, scalarFloatSwarmSize, 1));
	gpuErrchk(cudaMemset(d_functionals, 0, scalarFloatSwarmSize));

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	size_t texture_offset = 0;
	gpuErrchk(cudaBindTexture2D(&texture_offset, functional_texture, d_functionals, channelDesc, numberOfParticles, 1, pitch)); 
    functional_texture.normalized = true; 
    functional_texture.addressMode[0] = cudaAddressModeWrap;

	// --- Initializing the current personal best fitnesses (numberOfParticles)
	gpuErrchk(cudaMalloc((void**)&d_personal_best_functional, scalarFloatSwarmSize));
	gpuErrchk(cudaMemset(d_personal_best_functional, 0, scalarFloatSwarmSize));

	scalarIntSwarmSize = numberOfParticles * sizeof(unsigned int);
	
	// --- Allocation of the local best ids, namely, the indices (for all particles) of the best neighbour (for the ring topology) (numberOfParticles)
	gpuErrchk(cudaMalloc((void**)&d_localBestIDs, scalarIntSwarmSize));
	gpuErrchk(cudaMemset(d_localBestIDs, 0, scalarIntSwarmSize));

	// --- Allocation of the update flags saying to each particle whether to update their personal best (numberOfParticles)
	gpuErrchk(cudaMalloc((void**)&d_to_be_updated, scalarIntSwarmSize));
	gpuErrchk(cudaMemset(d_to_be_updated, 0, scalarIntSwarmSize));

	// --- Allocation of the CUDA random states
	gpuErrchk(cudaMalloc((void **)&devStates, numberOfParticles * actualParticleSize * sizeof(curandState)));

	// --- Setting up random generator states
	init_pseudorandom_generator<<<numberOfParticles, actualParticleSize>>>(devStates, time(NULL));
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	h_globalBestPosition = (float *)malloc(numberOfUnknowns * sizeof(float));

}
	
	
/************************************/
/* UPDATE OF THE PARTICLE POSITIONS */
/************************************/
__global__ void g_positionsUpdate(const unsigned int * __restrict__ d_localBestIDs, 
	                                     const unsigned int * __restrict__ d_to_be_updated, float * __restrict__ d_positions, 
										 float * __restrict__ d_best_personal_positions, float * __restrict__ d_velocities, 
										 curandState * __restrict__ devStates){

	// --- blockIdx.x addresses the particle inside the swarm
	// --- threadIdx.x addresses the dimension (unknown) within the particle

	// --- gridDim.x represents the NUMBER_OF_PARTICLES
	// --- blockDim.x represents the actualParticleSize (PROBLEM_DIMENSIONS aligned up to a multiple of 16)

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ unsigned int s_update;		// --- Best position update flag (tells if the specific particle is the global best or not)
	__shared__ unsigned int s_bestID; 		// --- Beginning of the local best individual

	//The first thread load the best position s_update flag and the index of the local best individual
	if(threadIdx.x == 0){
		s_update = d_to_be_updated[blockIdx.x];
		s_bestID = d_localBestIDs[blockIdx.x] * blockDim.x;
	}
	__syncthreads();

	float pos		= d_positions[tid];			// --- Current position
	float bestPos	= d_best_personal_positions[tid];		// --- Current particle best position
	float vel		= d_velocities[tid];		// --- Current velocity

	// --- Load the 2 pseudo random numbers needed to update position and velocity
	float R1, R2;
	R1 = curand_uniform(&devStates[tid]);
	R2 = curand_uniform(&devStates[tid]);

	if (s_update){
		bestPos = pos;										// --- Update local best position
		d_best_personal_positions[tid] = bestPos;			// --- Update the local best position to global memory
	}

	__threadfence(); 
	
	vel *= W;														// --- Apply inertia factor
	vel += C1 * R1 * (bestPos - pos);								// --- Add to the velocity the Cognitive Contribution
	
	vel += C2 * R2 * (d_best_personal_positions[s_bestID+threadIdx.x] - pos);	// --- Add to the velocity the Social Contribution

	d_velocities[tid] = vel;										// --- Update velocity to global memory

	pos += vel;														// --- Position update

	// --- Clamping the position to the actual search space
	pos = min(pos, c_maxValues[threadIdx.x]);
	pos = max(pos, c_minValues[threadIdx.x]);

	d_positions[tid] = pos;											// --- Update position to global memory
}

/***********************************/
/* FIND THE LOCAL BEST PARTICLE ID */
/***********************************/
__global__ void local_best_update(const float * __restrict__ d_functionals, float * __restrict__ d_personal_best_functional, 
	                              unsigned int * __restrict__ d_localBestIDs, unsigned int * __restrict__ d_to_be_updated, const unsigned int generationNumber){

	// --- particleID addresses the particle inside the swarm
	// --- blockDim.x * gridDim.x is greater or equal to all the particles
	int particleID = threadIdx.x + blockIdx.x * blockDim.x;

	if (particleID < numberOfParticles) {
	
		float *local_functionals = (float *)malloc((2 * RADIUS + 1) * sizeof(float));

		// --- Load the functional values from global memory
		for (int i = 0; i < 2 * RADIUS + 1; i++) local_functionals[i] = tex1D(functional_texture, (float)(particleID + 0.5 + (i - RADIUS)) / (float)numberOfParticles);

		thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(local_functionals);
		thrust::device_ptr<float> min_ptr = thrust::min_element(thrust::seq, dev_ptr, dev_ptr + 2 * RADIUS + 1);
		float min_value = min_ptr[0];

		// --- Writes the local best-ID to global memory
		d_localBestIDs[particleID] = (particleID + (&min_ptr[0] - &dev_ptr[0]) - RADIUS + numberOfParticles) % numberOfParticles;

		if (generationNumber > 0) {
			// --- Possibly update of both the best fitness value and the best position update flag
			unsigned int update = local_functionals[RADIUS] < d_personal_best_functional[particleID];
			d_to_be_updated[particleID] = update;
			if (update) d_personal_best_functional[particleID] = local_functionals[RADIUS]; 
		}
		else
			// --- Initially, the best personal fitness value is the first fitness value
			d_personal_best_functional[particleID] = local_functionals[RADIUS];
	}	
}

/*****************************************************************************/
/* FUNCTION TO FIND THE GLOBAL BEST PARTICLE AFTER COMPLETING THE ITERATIONS */
/*****************************************************************************/
void h_findGlobalBest(float * __restrict__ h_globalBestFitness, unsigned int * __restrict__ h_globalBestID, 
				      float * __restrict__ d_personal_best_functional){

	thrust::device_ptr<float> dp = thrust::device_pointer_cast(d_personal_best_functional);
	thrust::device_ptr<float> pos = thrust::min_element(dp, dp + numberOfParticles);

	*h_globalBestID = thrust::distance(dp, pos);

	gpuErrchk(cudaMemcpy(h_globalBestFitness, &d_personal_best_functional[*h_globalBestID], sizeof(float), cudaMemcpyDeviceToHost));
}

/******************************/
/* TRANSFORM REDUCTION KERNEL */
/******************************/
__global__ void CostFunctionalCalculation(const float * __restrict__ indata, float * __restrict__ outdata) {
	
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// --- Specialize BlockReduce for type float. 
	typedef cub::BlockReduce<float, numberOfUnknowns> BlockReduceT;

	__shared__ typename BlockReduceT::TempStorage  temp_storage;

	float result;
	if(tid < (numberOfParticles * numberOfUnknowns) ) result = BlockReduceT(temp_storage).Sum(sphere(indata[tid]));

	if(threadIdx.x == 0) outdata[blockIdx.x] = result; 
  
	return;
}

/*************************/
/* OPTIMIZATION FUNCTION */
/*************************/
void h_PSO_Optimize(void)
{
	printf("Starting Optimization...\n");

	for (int i = 0; i < numberOfUnknowns; i++) DELTAVALS[i] = MAXVALS[i] - MINVALS[i];
	
	// --- Set up search space limits
	gpuErrchk(cudaMemcpyToSymbol(c_minValues,    &MINVALS,     numberOfUnknowns * sizeof(float), 0, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(c_maxValues,    &MAXVALS,     numberOfUnknowns * sizeof(float), 0, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(c_deltaValues,  &DELTAVALS,   numberOfUnknowns * sizeof(float), 0, cudaMemcpyHostToDevice));

	// --- Particles initialization
	particles_initialization<<<numberOfParticles, actualParticleSize>>>(d_positions, d_best_personal_positions, d_velocities, devStates);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// --- Set the update flags to zero
	gpuErrchk(cudaMemset(d_to_be_updated, 0, numberOfParticles * sizeof(unsigned int)));

	// --- First fitnesses evaluation
	CostFunctionalCalculation<<<numberOfParticles, numberOfUnknowns>>>(d_positions, d_functionals);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	unsigned int generationNumber = 0;
	
	// --- First time local bests update
	local_best_update<<<iDivUp(numberOfParticles, BLOCKSIZE_LOCAL_BEST), BLOCKSIZE_LOCAL_BEST>>>(d_functionals, d_personal_best_functional, d_localBestIDs, d_to_be_updated, generationNumber);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// --- GENERATIONS
	for(generationNumber = 1; generationNumber < numberOfGenerations; ++generationNumber){

		// --- PositionS Update
		g_positionsUpdate<<<numberOfParticles, actualParticleSize>>>(d_localBestIDs, d_to_be_updated, d_positions, d_best_personal_positions, d_velocities, devStates);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Fitness evaluation
		CostFunctionalCalculation<<<numberOfParticles, numberOfUnknowns>>>(d_positions, d_functionals);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Local bests update
		local_best_update<<<iDivUp(numberOfParticles, BLOCKSIZE_LOCAL_BEST), BLOCKSIZE_LOCAL_BEST>>>(d_functionals, d_personal_best_functional, d_localBestIDs, d_to_be_updated, generationNumber);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

	}

	// --- Global best determination
	unsigned int h_globalBestID;
	h_findGlobalBest(&h_globalBestFitness, &h_globalBestID, d_personal_best_functional);

	gpuErrchk(cudaMemcpy(h_globalBestPosition, d_best_personal_positions + h_globalBestID * actualParticleSize, numberOfUnknowns * sizeof(float), cudaMemcpyDeviceToHost));

	printf("Number of partices = %d\n Number of unknowns = %d\n Number of iterations = %d\n Minimum found = %e\n", numberOfParticles, numberOfUnknowns, numberOfGenerations, h_globalBestFitness);

	for (int k=0; k<numberOfUnknowns; k++) printf("%i %f\n", k, h_globalBestPosition[k]);
}

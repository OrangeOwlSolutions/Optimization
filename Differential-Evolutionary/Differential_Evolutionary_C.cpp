#include <stdio.h>
#include <time.h>
#include <fstream>

#define pi 3.14159265358979f

// --- REFERENCES
//     [1] R. Storn and K. Price, “Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces,” 
//     Journal of Global Optimization, vol. 11, no. 4, pp. 341–359, 1997

//     [2] Lucas de P. Veronese and Renato A. Krohling, “Differential Evolution Algorithm on the GPU with C-CUDA,” 
//     Proc. of the IEEE Congress on Evolutionary Computation, Barcelona, Spain, Jul. 18-23, 2010, pp. 1-7.

// Conventions: the index j addresses the population member while the index i addresses the member component
//              the homologous host and device variables have the same name with a "h_" or "d_" prefix, respectively
//				the __host__ and __device__ functions pointer parameters have the same name for comparison purposes. it is up to the caller to use 
//				host or device pointers, as appropriate

/********************************/
/* INITIALIZE POPULATION ON CPU */
/********************************/
void initialize_population_CPU(float * __restrict pop, const float * __restrict minima, const float * __restrict maxima, 
	                           const unsigned long seed, const int D, const int Np) {

	srand(seed);

	for (int j=0; j<Np; j++)
		for (int i=0; i<D; i++)
			pop[j*D+i] = (maxima[i] - minima[i]) * ((float)rand()/(float)RAND_MAX) + minima[i];
}

/****************************************/
/* EVALUATION OF THE OBJECTIVE FUNCTION */
/****************************************/
float functional(const float * __restrict x, const int D) {

	float sum = 0.f;

	// --- De Jong function
	//for (int i=0; i<D; i++) sum = sum + x[i] * x[i];
	// --- Rosenbrock's saddle
	sum = 0.f;
	for (int i=1; i<D; i++) sum = sum + 100.f * (x[i] - x[i-1] * x[i-1]) * (x[i] - x[i-1] * x[i-1]) + (x[i-1] - 1.f) * (x[i-1] - 1.f);

	return sum;
}

/********************************/
/* POPULATION EVALUATION ON CPU */
/********************************/
void evaluation_CPU(const int Np, const int D, const float * __restrict pop, float * __restrict fobj) {

	for (int j=0; j<Np; j++) fobj[j] = functional(&pop[j*D], D);
}

/**********************************************************/
/* GENERATE MUTATION INDICES AND CROSS-OVER VALUES ON CPU */
/**********************************************************/
void generate_mutation_indices_and_cross_over_values_CPU(int * __restrict mutation, float * __restrict Rand, const int Np, const unsigned long seed) {

	int a, b, c;
	
	for(int j=0; j<Np; j++) {
		do a=Np*((float)rand()/(float) RAND_MAX); while(a==j);
		do b=Np*((float)rand()/(float) RAND_MAX); while(b==j||b==a);
		do c=Np*((float)rand()/(float) RAND_MAX); while(c==j||c==a||c==b);
		mutation[j*3]=a;
		mutation[j*3+1]=b;
		mutation[j*3+2]=c;
	}

	for(int j=0; j<Np; j++) Rand[j]=((float)rand()/(float) RAND_MAX);
}

/**********************************/
/* GENERATION OF A NEW POPULATION */
/**********************************/
void generation_new_population_CPU(float *pop, int Np, int D, float *npop, float F, float CR, float *rand, int *mutation, float* minimum, float* maximum) {

	int a, b, c;

	for (int j=0; j<Np; j++) {

		// --- Mutation indices
		int a=mutation[j*3];
		int b=mutation[j*3+1];
		int c=mutation[j*3+2];
		
		for (int i=0; i<D; i++) {

			// --- Mutation and crossover
			if(rand[j]<CR)	{
				npop[j*D+i] = pop[a*D+i]+F*(pop[b*D+i]-pop[c*D+i]);
			}
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
			if		(npop[j*D+i] > maximum[i])	npop[j*D+i] = maximum[i];
			else if	(npop[j*D+i] < minimum[i])	npop[j*D+i] = minimum[i];
		}
	}
}

/*******************************/
/* POPULATION SELECTION ON CPU */
/*******************************/
void selection_and_evaluation_CPU(const int Np, const int D, float * __restrict pop, const float * __restrict npop, float * __restrict fobj) {

	float nfobj;

	for (int j=0; j<Np; j++) {

		nfobj = functional(&npop[j*D], D);

		if (nfobj < fobj[j]) { 
			for (int i=0; i<D; i++) { 
				pop[j*D+i]	= npop[j*D+i];
			}
			fobj[j]		= nfobj;
		}
	}
}

/***********************/
/* FIND MINIMUM ON CPU */
/***********************/
void find_minimum_CPU(const int N, const float * __restrict t, float * __restrict minval, int * __restrict index) {
    
	minval[0]	= t[0];
	index[0]	= 0;

    for (int i = 1; i<N; i++) {
		
		if(t[i] < minval[0]) {
			minval[0]	= t[i]; 
			index[0]	= i;
		}
	}
    
}

/********/
/* MAIN */
/********/
int main()
{
	// --- Number of individuals in the population (Np >=4 for mutation purposes)
	int			Np		= 100;  
	// --- Dimensionality of each individual (number of unknowns)
	int			D		= 5;
	// --- Mutation factor (0 < F <= 2). Typically chosen in [0.5, 1], see Ref. [1]
	float		F		= 1.f;
	// --- Maximum number of generations
	int			Gmax	= 3000;
	// --- Crossover constant (0 < CR <= 1)
	float		CR		= 0.3f;

	// --- Mutually different random integer indices selected from {1, 2, … ,Np}
	int *h_mutation,			// --- Host side mutation vector
		*h_best_index;			// --- Host side current optimal member index

	float *h_pop,				// --- Host side population
	*h_npop,					// --- Host side new population (trial vectors)
	*h_Rand,					// --- Host side crossover rand vector (uniformly distributed in (0,1))
	*h_fobj,					// --- Host side objective function value
	*h_best,					// --- Host side population best value history
	*h_maxima,					// --- Host side maximum constraints vector
	*h_minima;					// --- Host side minimum constraints vector

	// --- Host side memory allocations
	h_pop				= (float*)malloc(D*Np*sizeof(float));
	h_npop				= (float*)malloc(D*Np*sizeof(float));
	h_Rand				= (float*)malloc(Np*sizeof(float));
	h_best				= (float*)malloc(Gmax*sizeof(float));
	h_best_index		= (int*)malloc(Gmax*sizeof(int));
	h_fobj				= (float*)malloc(Np*sizeof(float));
	h_mutation			= (int*)  malloc(3*Np*sizeof(int));
	h_maxima			= (float*)malloc(D*sizeof(float));
	h_minima			= (float*)malloc(D*sizeof(float));

	// --- Set maxima and minima
	for (int i=0; i<D; i++) {
		h_maxima[i] =  2.f;
		h_minima[i] = -2.f;
	}
	
	// --- Initialize popultion
	initialize_population_CPU(h_pop, h_minima, h_maxima, time(NULL), D, Np);

	// --- Evaluate population
	evaluation_CPU(Np, D, h_pop, h_fobj);

	int a, b, c;
	for(int i=0;i<Gmax;i++) {
		
		// --- Generate mutation indices and cross-over uniformly distributed random vector
		generate_mutation_indices_and_cross_over_values_CPU(h_mutation, h_Rand, Np, time(NULL));
	
		// --- Generate new population
		generation_new_population_CPU(h_pop, Np, D, h_npop, F, CR, h_Rand, h_mutation, h_minima, h_maxima);

		// --- Select new population and evaluate it
		selection_and_evaluation_CPU(Np, D, h_pop, h_npop, h_fobj);

		find_minimum_CPU(Np, h_fobj, &h_best[i], &h_best_index[i]);
		
		printf("Host.   Iteration: %i; best member value: %f: best member index: %i\n", i, h_best[i], h_best_index[i]);

	}

	for (int i=0; i<D; i++) printf("Variable nr. %i = %f\n", i, h_pop[h_best_index[Gmax-1]*D+i]);

	getchar();
	
	return 0;
}


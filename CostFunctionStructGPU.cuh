/***********************************/
/* COST FUNCTION STRUCT - GPU CASE */
/***********************************/
// --- Rosenbrock function struct for thrust reduction
struct CostFunctionStructGPU{
template <typename Tuple>
	__host__ __device__ float operator()(Tuple a) {

		float temp1 = (thrust::get<1>(a) - thrust::get<0>(a) * thrust::get<0>(a));
		float temp2 = (thrust::get<0>(a) - 1.f);
	
		return 100.f * temp1 * temp1 + temp2 * temp2;
	}
};


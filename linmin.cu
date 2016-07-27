#include <thrust\device_vector.h>
#include <thrust\inner_product.h>
#include <thrust\functional.h>

#include "CostFunctionStructGPU.cuh"
#include "mnbrak.cuh"
#include "dbrent.cuh"

/****************************/
/* DIRECTION UPDATE FUNCTOR */
/****************************/
struct p_update_functor
{
    const float a;

    p_update_functor(float _a) : a(_a) {}

    __host__ __device__ float operator()(const float& x) const { return a * x; }
};

/****************************/
/* UNKNKOWNS UPDATE FUNCTOR */
/****************************/
struct x_update_functor
{
    x_update_functor() {}

    __host__ __device__ float operator()(const float& x, const float& y) const { return x + y; }
};

/*************************/
/* ABSOLUTE VALUE STRUCT */
/*************************/
template <typename T>
struct absolute_value {
	__host__ __device__
		T operator()( const T& x ) const { return fabs(x); }
};

/******************/
/* LINMIN ROUTINE */
/******************/
void linmin(const thrust::device_ptr<float> dev_ptr_x, const thrust::device_ptr<float> dev_ptr_p, const int N) {

	// --- d_p									Search direction
	// --- d_x									Unknowns (input - output)
	// --- N									Number of unknowns
	// --- itmax								Maximum number of iterations
	
	// --- Bracketing tolerance
	const float TOL = 2.0e-4;

	const float lsm = thrust::transform_reduce(dev_ptr_p, dev_ptr_p + N, absolute_value<float>(), 0.f, thrust::maximum<float>());

	float ax = 0.0f;
	float xx = .1e-2/(lsm+1.e-8);
	float bx = .2e-2/(lsm+1.e-8);

	mnbrak(dev_ptr_x, dev_ptr_p, &ax, &xx, &bx, N);

	float xmin;
	dbrent(dev_ptr_x, dev_ptr_p, ax, xx, bx, TOL, &xmin, N);

	thrust::transform(dev_ptr_p, dev_ptr_p + N, dev_ptr_p, p_update_functor(xmin));
	thrust::transform(dev_ptr_x, dev_ptr_x + N, dev_ptr_p, dev_ptr_x, x_update_functor());

}

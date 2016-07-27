#include <math.h>

#include <thrust\device_vector.h>

#include "Utilities.cuh"
#include "FuncGrad.cuh"

#define GOLD 1.618034
#define GLIMIT 100.0f
#define TINY 1.0e-20
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : -fabs(a))
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

/*******************************************************************************/
/* CLASSICAL NUMERICAL RECIPES' MNBRAK FUNCTION ADAPTED TO PARALLEL PROCESSING */
/*******************************************************************************/
void mnbrak(const thrust::device_ptr<float> dev_ptr_x, const thrust::device_ptr<float> dev_ptr_p, float *ax, float *bx, float *cx, const int N) {
	
	float ulim, u, r, q, fu, dum;

	float *d_xt;				gpuErrchk(cudaMalloc((void**)&d_xt,			N * sizeof(float)));	thrust::device_ptr<float> dev_ptr_xt			= thrust::device_pointer_cast(d_xt);	
	
	float fa = f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, *ax, N);
	float fb = f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, *bx, N);
	if (fb > fa) {
		SHFT(dum, *ax, *bx, dum);
		SHFT(dum,  fb,  fa, dum);
	}
	*cx = (*bx) + GOLD * (*bx - *ax);
	float fc = f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, *cx, N);
	while (fb > fc) {
		r		= (*bx-*ax) * (fb - fc);
		q		= (*bx-*cx) * (fb - fa);
		u		= (*bx) - ((*bx - *cx) * q - (*bx - *ax) * r) / (2.f * SIGN(MAX(fabs(q - r), TINY), q - r));
		ulim	= (*bx) + GLIMIT * (*cx - *bx);
		if ((*bx - u) * (u - *cx) > 0.0f) {
			fu = f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, u, N);
			if (fu < fc) {
				*ax = (*bx);
				*bx = u;
				fa  = fb;
				fb  = fu;
				return;
			} else if (fu > fb) {
				*cx = u;
				fc  = fu;
				return;
			}
			u  = (*cx) + GOLD * (*cx - *bx);
			fu = f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, u, N);
		} else if ((*cx - u) * (u - ulim) > 0.0f) {
			fu = f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, u, N);
			if (fu < fc) {
				SHFT(*bx, *cx,  u, *cx + GOLD * (*cx - *bx));
				SHFT(fb,   fc, fu, f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, u, N));
			}
		} else if ((u - ulim) * (ulim - *cx) >= 0.0) {
			u  = ulim;
			fu = f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, u, N);
		} else {
			u  = (*cx) + GOLD * (*cx - *bx);
			fu = f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, u, N);
		}
		SHFT(*ax, *bx, *cx,  u);
		SHFT( fa,  fb,  fc, fu);
	}
}

#undef GOLD
#undef GLIMIT
#undef TINY
#undef MAX
#undef SIGN
#undef SHFT

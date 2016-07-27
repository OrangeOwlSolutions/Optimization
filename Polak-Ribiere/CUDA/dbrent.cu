#include <thrust\device_vector.h>

#include "Utilities.cuh"
#include "FuncGrad.cuh"

#include <math.h>

#define ITMAX 100
#define ZEPS 1.0e-10
#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : -fabs(a))
#define MOV3(a,b,c, d,e,f) (a)=(d);(b)=(e);(c)=(f);
 
/*******************************************************************************/
/* CLASSICAL NUMERICAL RECIPES' DBRENT FUNCTION ADAPTED TO PARALLEL PROCESSING */
/*******************************************************************************/
void dbrent(const thrust::device_ptr<float> dev_ptr_x, const thrust::device_ptr<float> dev_ptr_p, float ax, float cx, float bx, float tol, float *xmin, const int N) {

	int		iter, ok1, ok2;
	float	d , d1, d2, du,   dv,   dw,   dx, e=0.0f;
	float	fu, fv, fw, fx, olde, tol1, tol2, u, u1, u2, v, w, x, xm;

	float *d_xt;				gpuErrchk(cudaMalloc((void**)&d_xt,			N * sizeof(float)));	thrust::device_ptr<float> dev_ptr_xt			= thrust::device_pointer_cast(d_xt);	
	
	float a = (ax < cx ? ax : cx);
	float b = (ax > cx ? ax : cx);
	x = w = v = bx;
	fw = fv = fx = f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, x, N);
	dw = dv = dx = df1dim(d_xt, dev_ptr_p, u, N);
	for (iter=1; iter <= ITMAX; iter++) {
		xm		= 0.5f * (a + b);
		tol1	= tol  * fabs(x) + ZEPS;
		tol2	= 2.f  * tol1;
		if (fabs(x - xm) <= (tol2 - 0.5f * (b - a))) *xmin = x;
		if (fabs(e) > tol1) {
			d1	= 2.0f * (b - a);
			d2	= d1;
			if (dw != dx)  d1 = (w - x) * dx / (dx - dw);
			if (dv != dx)  d2 = (v - x) * dx / (dx - dv);
			u1	 = x + d1;
			u2	 = x + d2;
			ok1  = (a - u1) * (u1 - b) > 0.f && dx * d1 <= 0.f;
			ok2  = (a - u2) * (u2 - b) > 0.f && dx * d2 <= 0.f;
			olde = e;
			e    = d;
			if (ok1 || ok2) {
				if (ok1 && ok2) d=(fabs(d1) < fabs(d2) ? d1 : d2);
				else if (ok1) d = d1; else d = d2;
				if (fabs(d) <= fabs(0.5f * olde)) {
					u=x+d;
					if (u - a < tol2 || b - u < tol2) d = SIGN(tol1, xm - x);
				} 
				else { d = 0.5f * (e = (dx >= 0.f ? a - x : b - x)); }
			} 
			else { d = 0.5f * (e = (dx >= 0.f ? a - x : b - x)); }
		} 
		else { d = 0.5f * (e = (dx >= 0.f ? a - x : b - x)); }
		if (fabs(d) >= tol1) {
			u	= x + d;
			fu	= f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, u, N);
		} 
		else {
			u	= x + SIGN(tol1, d);
			fu	= f1dim(dev_ptr_x, dev_ptr_p, dev_ptr_xt, u, N);
			if (fu > fx) *xmin = x;
		}
		du = df1dim(d_xt, dev_ptr_p, u, N);
		if (fu <= fx) {
			if (u >= x) a = x; else b = x;
			MOV3(v, fv, dv, w, fw, dw)
			MOV3(w, fw, dw, x, fx, dx)
			MOV3(x, fx, dx, u, fu, du)
		} else {
			if (u < x) a = u; else b = u;
			if (fu <= fw || w == x) {
				MOV3(v, fv, dv, w, fw, dw)
				MOV3(w, fw, dw, u, fu, du)
			} 
			else if (fu < fv || v == x || v == w) MOV3(v,fv,dv, u,fu,du)
		}
	}
}

#undef ITMAX
#undef ZEPS
#undef SIGN
#undef MOV3

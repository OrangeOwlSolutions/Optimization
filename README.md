# CUDA Cost Functional Calculation
CUDA Calculation of typical optimization cost functionals for non-linear optimization using thrust::transform_reduce and a customized version of transform_reduce

Many cost functionals are expressed in the form of summations of a certain number of terms. Examples are the 
```Sphere function```, the ```Rosenbrock function``` and the ```Styblinski-Tang function```, see 

http://en.wikipedia.org/wiki/Test_functions_for_optimization

In all those cases, the evaluation of the cost function can be performed by a reduction, or better, a transformation followed by 
a reduction.

CUDA Thrust has ```thrust::transform_reduce``` which can surely serve the scope, but of course you can set up your own 
transformation + reduction routines.

In this project, we provide an example on how you can compute the Rosenbrock functional using either CUDA Thrust or a customized 
version of the reduction routine offered by the CUDA examples. In the latter case, a pointer to a ```__device__``` transformation 
function is passed to the customized ```transform_reduce``` function, if the ```EXTERNAL``` keyword in the customized
```transform_reduce.cu``` file is defined, or the transformation function is defined and compiled in the compilation unit of the 
customized ```transform_reduce``` routine.

Usage:
```
If you define the EXTERNAL keyword in the transform_reduce.cu file, then define the transformation __device__
function in the compilation unit where you use the customized transform_reduce function as, for example,

// --- Transformation function
__device__ __forceinline__ float transformation(const float * __restrict__ x, const int i) { return (100.f * (x[i+1] - x[i] * x[i]) * (x[i+1] - x[i] * x[i]) + (x[i] - 1.f) * (x[i] - 1.f)) ; }
// --- Device-side function pointer
__device__ pointFunction_t dev_pfunc = transformation;

Otherwise, define the transformation __device__ function in the transform_reduce.cu compilation unit.
```

The files ```TimingCPU.cpp```, ```TimingCPU.h```, ```TimingGPU.cu``` and ```TimingGPU.cuh``` are omitted and mantained at

https://github.com/OrangeOwlSolutions/Timing

The files ```Utilities.cu``` and ```Utilities.cuh``` are omitted and mantained at

https://github.com/OrangeOwlSolutions/CUDA_Utilities

Please, note that separate compilation is required.

Some performance results on a Kepler K20c card for the non-EXTERNAL case:

```
N =   90000       Thrust = 0.055ms       Customized = 0.059ms
N =  900000       Thrust = 0.67ms        Customized = 0.14ms
N = 9000000       Thrust = 0.85ms        Customized = 0.87ms
```

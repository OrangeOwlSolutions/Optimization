# Gradient Descent Optimization in CUDA
Gradient Descent (Steepest Descent) Optimization in CUDA and C/C++

*Gradient Descent* (AKA *steepest descent*) aims at finding a local minimum of a multivariate function ```F(x)``` by taking steps 
proportional to the negative of the gradient of ```F(x)``` at the current point. The update rule is the following:

![alt tag](http://upload.wikimedia.org/math/8/b/0/8b0e3f1c41429f48f4788cfee9fe57ee.png)

where the step size ```gamma_n``` is allowed to change at every step and can be determined, for example, by line searches.

Implementing the above update rule in CUDA is pretty easy. In this project, we provide a full example using the *Rosenbrock function* as the cost functional to be optimized, exploiting the analytical gradient, and considering a constant value for the step size through the iterations (namely, ```gamma_n = gamma```). The ```Utilities.cu``` and ```Utilities.cuh``` files are mantained at 

https://github.com/OrangeOwlSolutions/CUDA_Utilities

and omitted here. The example implements the CPU as well as the GPU approach.

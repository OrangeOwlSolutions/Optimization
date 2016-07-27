function [f xt] = f1dim(lambda, xn, p, costfunctional)
	
xt = xn + lambda * p;

f = costfunctional(xt);

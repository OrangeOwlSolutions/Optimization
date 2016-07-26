function [y, g] = quadratic(x)

y = sum(abs(x).^2);

g = 2 * x;

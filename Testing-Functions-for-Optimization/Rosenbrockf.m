function [f, g] = Rosenbrockf(x)

N = length(x);

f = sum(100 * (x(2 : N) - x(1 : N - 1).^2).^2 + (x(1 : N - 1) - 1).^2);

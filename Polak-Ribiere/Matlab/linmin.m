% --- Line minimization ... see Numerical Recipes

function [x p] = linmin(x, p, itmax, costfunctional, grad_costfunctional)

% --- p                                 Search direction
% --- x                                 Unknowns (input - output)
% --- itmax                             Maximum number of iterations

% --- Bracketing tolerance
tol     = 1.e-8;

lsm     = max(abs(p));
disp('Prima di mnbrak')
ax      = 0.;
xx      = .1e-2/(lsm+1.e-10);
bx      = .2e-2/(lsm+1.e-10);
% xx      = 0;
% bx      = 1;

[ax bx cx] = mnbrak(ax, xx, bx, x, p, costfunctional);

lambdamin = dbrent(ax, cx, bx, tol, x, p, itmax, costfunctional, grad_costfunctional);

p = lambdamin * p;
x = x + p;

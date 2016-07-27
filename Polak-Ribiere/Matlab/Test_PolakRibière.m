clear all
close all
clc

N           = 2;              % --- Number of unknowns
itmax       = 10000;          % --- Maximum number of iterations
itmaxlinmin = 10;             % --- Maximum number of linmin iterations
Ftol        = 1.e-4;          % --- Functional change tolerance (in dB)

xstart = 2 * ones(1, N);

% --- Quadratic and Booth require very few iterations. Otherwise NaN since
% the functional becomes very small and produces underfloors

% x_PolakRibiere = PolakRibiere(xstart, itmax, itmaxlinmin, Ftol, @quadraticf, @quadraticg)
x_PolakRibiere = PolakRibiere(xstart, itmax, itmaxlinmin, Ftol, @Rosenbrockf, @Rosenbrockg)
% x_PolakRibiere = PolakRibiere(xstart, itmax, itmaxlinmin, Ftol, @Bealesf, @Bealesg);
% x_PolakRibiere = PolakRibiere(xstart, itmax, itmaxlinmin, Ftol, @Boothf, @Boothg);

% x_Matlab = fminunc(@quadratic,  xstart);
x_Matlab = fminunc(@Rosenbrock,  xstart);
% x_Matlab = fminunc(@Beales,     xstart);
% x_Matlab = fminunc(@Booth,     xstart);


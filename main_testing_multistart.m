clear all
close all
clc
warning off

N               = 10000;
gammap          = 0.95;
maxiter         = 10000;
maxiterlocal    = 100;
Constrained     = 1;    % --- 0 if unconstrained (fminunc) / 1 if constrained optimization (fmincon)

% --- Quadratic
% minvalues       = [-1 -1 -1];
% maxvalues       = [ 1  1  1];
% options = optimset('UseParallel', 'never', 'Disp', 'iter', 'GradObj', 'on', 'DerivativeCheck','on');
% [GlobalOptimum OptFuncVal] = MultistartOptimizer(minvalues, maxvalues, N, gammap, maxiter, maxiterlocal, options, Constrained, @quadratic)

% --- Beales
% minvalues       = [-4.5 -4.5];
% maxvalues       = [ 4.5  4.5];
% options = optimset('UseParallel', 'never', 'Disp', 'iter', 'GradObj', 'on');
% [GlobalOptimum OptFuncVal] = MultistartOptimizer(minvalues, maxvalues, N, gammap, maxiter, maxiterlocal, options, Constrained, @Beales)

% --- Booth
% minvalues       = [-10 -10];
% maxvalues       = [ 10  10];
% % options = optimset('UseParallel', 'never', 'Disp', 'iter', 'GradObj', 'on',  'DerivativeCheck','on');
% options = optimset('UseParallel', 'never', 'Disp', 'iter', 'GradObj', 'on');
% [GlobalOptimum OptFuncVal] = MultistartOptimizer(minvalues, maxvalues, N, gammap, maxiter, maxiterlocal, options, Constrained, @Booth)

% --- Rosenbrock
minvalues       = [ 0.5  0.5  0.5  0.5];
maxvalues       = [ 1.5  1.5  1.5  1.5];
% options = optimset('UseParallel', 'never', 'Disp', 'iter', 'GradObj', 'on', 'DerivativeCheck','on');
options = optimset('UseParallel', 'never', 'Disp', 'iter', 'GradObj', 'on');
[GlobalOptimum OptFuncVal] = MultistartOptimizer(minvalues, maxvalues, N, gammap, maxiter, maxiterlocal, options, Constrained, @Rosenbrock)



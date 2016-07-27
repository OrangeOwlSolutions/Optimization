% Given a function costfunctional and its derivative function grad_costfunctional, and given a bracketing triplet of abscissas ax,
% bx, cx [such that ax < bx < cx, and f(bx) < f(ax) and f(bx) < f(cx), tipically the output of mnbrak], this routine isolates the 
% minimum to a fractional precision of about tol using a modification of Brent’s method that uses derivatives. The abscissa of the 
% minimum is returned as lambdamin, and the minimum function value is returned as out.

% --- The function f(lambda) is a contraction for f(x + lambda * p)

function [lambdamin out] = dbrent(ax, bx, cx, tol, x, p, itmax, costfunctional, grad_costfunctional)
	
% --- itmax         number of iterations

zeps = 1.e-10;

% --- a and b are fixed as the extremes of the bracketing interval so that
% a < b
a       = min([ax cx]);
b       = max([ax cx]);
% --- lambda, lambda_1 and lambda_2 are the estimates of the minimum at the current iteration, at two iterations
% before, and at the last iteration, respectively. They are all initialized at initial estimate of the minimum
lambda      = bx;                
lambda_1    = lambda;
lambda_2    = lambda;
% --- xt = x + lambda * p
% --- flambda = f(x + lambda * p) is the functional value at the current
% estimate of the minimum
[flambda xt] = f1dim(lambda, x, p, costfunctional);
flambda_1    = flambda;
flambda_2    = flambda;
% --- dotlambda = <Gradf_xt,p>
dotlambda      = df1dim(xt, p, grad_costfunctional);
dotlambda_2    = dotlambda;
dotlambda_1    = dotlambda;

e       = 0.;

for iter=1:itmax
    
    % --- Bisection step
    xm      = 0.5 * (a + b);
    tol1    = tol * abs(lambda) + zeps;
    tol2    = 2. * tol1;
    
    % --- Convergence check. If the current minimum estimate lambda is
    % sufficiently close to the bisected point, then the minimum is
    % considered to be reached.
    if (abs(lambda - xm) <= (tol2 - 0.5 * (b - a))) 
        lambdamin = lambda;
        out  = flambda; 
        return;
    end
    
    if (abs(e) > tol1) 
        
        % --- Initialize these increment d’s to an out-of-bracket value
        d1 = 2. * (b - a);
        d2 = d1;
	
        % --- Secant method with one point
        if (dotlambda_1 ~= dotlambda) 
            d1 = (lambda_2 - lambda) * dotlambda / (dotlambda - dotlambda_1);
        end
        if (dotlambda_2 ~= dotlambda) 
            d2 = (lambda_1 - lambda) * dotlambda / (dotlambda - dotlambda_2);
        end
        
        % --- Which of these two estimates of d shall we take? We will insist that they be within
        %     the bracket, and on the side pointed to by the derivative at x:
        u1      = lambda + d1;
        u2      = lambda + d2;
	    ok1     = ((a - u1) * (u1 - b) > 0.) && (dotlambda * d1 <= 0.);
        ok2     = ((a - u2) * (u2 - b) > 0.) && (dotlambda * d2 <= 0.);
        olde    = e;
        e       = d;
        
        % --- Take only an acceptable d, and if both are acceptable, then take the smallest one.
        if (ok1 || ok2)
            if (ok1 && ok2)
                if (abs(d1) < abs(d2))
                    d = d1;
                else
                    d = d2;
                end
            else
                if (ok1)
                    d = d1;
                else
                    d = d2;
                end
            end
            if (abs(d) <= abs(0.5 * olde)) 
                u = lambda + d;
                if (((u - a) < tol2) || ((b - u) < tol2)) 
                    d = abs(tol1) * sign(xm - lambda);
                end
            else
                if (dotlambda >= 0.)
                    e = a - lambda;
                else
                    e = b - lambda;
                end
                d = 0.5 * e;
            end
        else
            if (dotlambda >= 0.)
                e = a - lambda;
            else
                e = b - lambda;
            end
            d = 0.5 * e;
        end
    else
        % --- If the scalar product between the gradient at the current
        % point and the search direction > 0, then set the bisection step d to bisect (a, lambda)
        % otherwise set the bisection step d to bisect (lambda, b)
        if (dotlambda >= 0.)
            e = a - lambda;
        else
            e = b - lambda;
        end
        d = 0.5 * e;
    end
    
    if (abs(d) >= tol1)
        % --- If the bisection step d is not too small, use d to perform a real bisection
        u = lambda + d;
    	[fu xt] = f1dim(u, x, p, costfunctional);
    else
        % --- If the bisection step d is not too small, use d to perform a real bisection
        u = lambda + abs(tol1) * sign(d);
    	[fu xt] = f1dim(u, x, p, costfunctional);
        % --- If the minimum step in the downhill direction takes us uphill, then we are done.
        if (fu > flambda)
            lambdamin = lambda;
            out  = flambda;
            return;
        end
    end
    du = df1dim(xt, p, grad_costfunctional);
    if (fu < flambda)
        % --- Enters here if the new trial minimum produces a lower
        % functional value
        if (u >= lambda)
            % --- Advance a to lambda if the current trial minimum is
            % larger than lambda
            a = lambda;
        else
            % --- Decrease b to lambda if the current trial minimim is
            % lower than lambda
            b = lambda;
        end
        % --- Saves the estimate of the minimum, its functional and the dot
        % product at two iterations before
        lambda_2       = lambda_1;
        flambda_2      = flambda_1;
        dotlambda_2    = dotlambda_1;
        % --- Saves the estimate of the minimum, its functional and the dot
        % product at the previous iteration
        lambda_1       = lambda;
        flambda_1      = flambda;
        dotlambda_1    = dotlambda;
        % --- Saves the estimate of the minimum, its functional and the dot
        % product at the current iteration
        lambda         = u;
        flambda        = fu;
        dotlambda      = du;
    else
        % --- Here fu > flambda
        if (u < lambda)
            a = u;
        else
            b = u;
        end
        if ((fu <= flambda_1) || (lambda_1 == lambda))
            % --- If the bisection point does not improve on the current
            % minimum estimate, but improves on the previous estimate (step -1), then
            % shift lambda_1 -> lambda_2 and u -> lambda_1
            lambda_2     = lambda_1;
            flambda_2    = flambda_1;
            dotlambda_2  = dotlambda_1;
            lambda_1     = u;
            flambda_1    = fu;
            dotlambda_1  = du;
        else
            if ((fu <= flambda_2) || (lambda_2 == lambda) || (lambda_2 == lambda_1))
                % --- If the bisection point does not improve on the current
                % minimum estimate, but improves on the previous estimate (step -1), then
                % shift u -> lambda_1
                lambda_2    = u;
                flambda_2   = fu;
                dotlambda_2 = du;
            end
        end
    end
end
disp('Too many iterations in dbrent')
lambdamin = lambda;
out = flambda;

% --- costfunctional          Cost functional to be optimized
function [GlobalOptimum OptFuncVal] = MultistartOptimizer(minvalues, maxvalues, N, gammap, maxiter, maxiterlocal, options, Constrained, costfunctional)

% --- N                       Number of sampling points introduced at each iteration
% --- gammap                  Percentage of the sampling points excluded at each iteration
% --- maxiter                 Maximum number of allowed multistart iterations
% --- maxiter                 Maximum number of allowed iterations for local optimization

Mopt = length(minvalues);           % --- Number of optimization variables
SamplingPoints = cell(Mopt);        % --- Current population of sampling points
LocalMinima = cell(Mopt);           % --- Found local minima
Func_val = [];                      % --- Functional values corresponding to the sampling points
Func_val_min = [];                  % --- Functional values corresponding to the local minima
flag = 0;                           % --- Exit flag
iter = 1;                           % --- Iteration counter
FoundMinima = 0;                    % --- Number of local minima

temp = zeros(1, Mopt);              % --- Temporary array

% --- Calculation of the prefactor needed for the computation of the critical distance
rho = 5;                            % --- Relevant to the calculation of the critical distance
prefactor_rk = 1;
for k=1:Mopt
    prefactor_rk = prefactor_rk * (maxvalues(k) - minvalues(k));
end
prefactor_rk = (1/sqrt(pi)) * (gamma(1+Mopt/2) * prefactor_rk * rho)^(1/Mopt);

options.MaxIter = maxiterlocal;

%%%%%%%%%%%%%%
% ITERATIONS %
%%%%%%%%%%%%%%

while ((flag == 0) && iter <= maxiter)
    
    % --- Introduce sampling points uniformly distributed in the search space
    for k=1:Mopt
        SamplingPoints{k} = [SamplingPoints{k} (maxvalues(k)-minvalues(k))*rand(1,N)+minvalues(k)];
    end
    
    % --- Calculate the functional values corresponding to the sampling points
    Func_val = zeros(1, length(SamplingPoints{1}));
    for p=1:length(SamplingPoints{1}),
        for q=1:Mopt,
            temp(q) = SamplingPoints{q}(p);
        end
        Func_val(p) = costfunctional(temp);
    end
    
    % --- Calculate the critical distance
    rk = prefactor_rk * (log(iter*N)/(iter*N))^(1/Mopt);
        
    % --- Sorts the functional values and the sampling points accordingly
    [Func_val,indices]=sort(Func_val);
    for k=1:Mopt
        SamplingPoints{k} = SamplingPoints{k}(indices);
    end
    
    % --- Determines the sampling points whose functional values are lower than gammap times the maximum functional value
    count=1;
    while (Func_val(count) <= gammap*Func_val(length(Func_val)));
        count=count+1;
    end
    count=count-1;
            
    % --- Explores all the current members of the population
    for pp=1:count,
        
        % --- Calculates the distance of the pp-th sampling point to all the other sampling points
        distance = zeros(size(SamplingPoints{1}));
        for k=1:Mopt
            distance = distance + (SamplingPoints{k}(pp) - SamplingPoints{k}).^2;
        end
        distance = sqrt(distance);
        
        % --- Filters the sampling points for which there exist at least one other sampling point spaced for less than rk and having a lower functional value  
        indices = find((distance <= rk) & (Func_val < Func_val(pp)));
            
        if (sum(indices) == 0)
            
            for q=1:Mopt,
                temp(q) = SamplingPoints{q}(pp);
            end
            if (Constrained == 1)
                [xstar,Func_val_opt]=fmincon(costfunctional,temp,[],[],[],[],minvalues,maxvalues,[],options);
            else
                [xstar,Func_val_opt]=fminunc(costfunctional,temp,options);
            end

            % --- If the iteration is the first one, then adds the found minimum
            if (iter==1)
                for q=1:Mopt,
                    LocalMinima{q} = [LocalMinima{q} xstar(q)];
                end
                Func_val_min=[Func_val_min Func_val_opt];
                % --- Updates the number of found local minima
                FoundMinima = FoundMinima + 1;
            else
                % --- Here iter > 1
                % --- Calculates the distance between the found minimum and all the other minima
                distance = 0.;
                for k=1:Mopt
                    distance = distance + (SamplingPoints{k}(pp) - SamplingPoints{k}).^2;
                end
                distance = sqrt(distance);
                % --- If the minimum has not been previously found, it is added to the local minima pool 
                if ((sum(find(distance == 0 ))==0))
                    for q=1:Mopt,
                        LocalMinima{q}(pp) = [LocalMinima{q}(pp) xstar(q)];
                    end
                    Func_val_min=[Func_val_min Func_val_opt];
                    % --- Updates the number of found local minima
                    FoundMinima = FoundMinima + 1;
                end
            end
        end        
    end
    
    iter=iter + 1;
    
    % --- Exit condition
    ExpectedCoveredUnknownSpace   = (count - FoundMinima - 1) * (count + FoundMinima) / (count * (count - 1));
    ExpectedNumberLocalMinima     = FoundMinima * (count - 1) / (count - FoundMinima - 2);
    flag = (ExpectedNumberLocalMinima < FoundMinima + 0.499) && ExpectedCoveredUnknownSpace >= 0.995;
end

% --- Retrieving the global optimizer
[OptFuncVal, index] = min(Func_val_min);
for q=1:Mopt,
    GlobalOptimum(q) = SamplingPoints{q}(index);
end

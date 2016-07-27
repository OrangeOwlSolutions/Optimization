function  x = PolakRibiere(x, itmax, itmaxlinmin, Ftol, costfunctional, grad_costfunctional)

% --- Ftol              Functional change tolerance
% --- itmax             Maximum number of allowed iterations	
% --- itmaxlinmin       Maximum number of linmin iterations

% --- Number of unknowns
N = length(x);

% --- Calculate starting function value and gradient
Func  = costfunctional(x);
Grad  = grad_costfunctional(x);

disp(strcat('Iteration = ', num2str(0), '; Functional value = ', num2str(10*log10(Func))))

% --- Initialize the value of the functional at the previous steps
Fold = Inf;

% --- Initialize the search direction
XI = -Grad;

% --- Initialize the iterations count
cont = 1;                              

% --- Initialize the exit flag
exitflag = 0;                   

%%%%%%%%%%%%%%
% ITERATIONS %
%%%%%%%%%%%%%%
while ((cont <= itmax) && (exitflag==0)),
    
    % --- Save the current value of the unknowns
    Old_x = x;

    %  --- Line minimization
%     [x XI] = linmin(x, XI, costfunctional, grad_costfunctional);
    x = linmin(x, XI, itmaxlinmin, costfunctional, grad_costfunctional);
   
    % --- Save the current value of the functional
    Fold = Func;

    % --- Save the current value of the gradient
    Old_Grad = Grad;

    Func  = costfunctional(x);
    Grad  = grad_costfunctional(x);
    
    disp(strcat('Iteration = ', num2str(cont), '; Functional value = ', num2str(10*log10(Func))))

    % --- If the new functional value is larger than that at the previous
    % step, then exit
    if (Func > Fold)
        x = Old_x;
        disp('Func > Fold')
        exitflag = 1;
    end
    
    % --- If the functional has a small change, then exit
    if (abs(10 * log10(abs(Func)) - 10 * log10(abs(Fold))) <= Ftol)
        exitflag = 1;
        disp('Small functional change')
    end
    
    if (exitflag == 0)

      gamma_PR=dot(Grad-Old_Grad,Grad)/dot(Old_Grad,Old_Grad);
      XI=-Grad+gamma_PR*XI;
      
      cont=cont+1;

   end   
   
end

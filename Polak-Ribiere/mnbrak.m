% Given a function costfunctional, and given distinct initial points ax and bx, this routine searches in
% the downhill direction (defined by the function as evaluated at the initial points) and returns
% new points ax, bx, cx that bracket a minimum of the function. The points ax, bx and cx are such that 
% the minimum is within ax and cx and in the proximity of bx. In other words, ax < bx < cx

function  [ax, bx, cx] = mnbrak(ax, bx, cx, x, p, costfunctional)
	
% --- p                                 Search direction

gold    = 1.618034;
glimit  = 100.;
tiny    = 1.e-20;

% --- It is assumed that fb < fa. If not, swap ax and bx
fa = f1dim(ax, x, p, costfunctional);
fb = f1dim(bx, x, p, costfunctional);
if (fb > fa)
    [ax bx] = swap(ax, bx);
    [fa fb] = swap(fa, fb);
end

% --- First guess for cx
cx = bx + gold * (bx - ax);
fc = f1dim(cx, x, p, costfunctional);

% --- Keep runnning until we bracket
while (fb > fc)
    
    % --- Compute u by parabolic extrapolation from ax, bx, cx. TINY is used to prevent any possible division by zero
    r       = (bx - ax) * (fb - fc);
    q       = (bx - cx) * (fb - fa);
    u       = bx - ((bx - cx) * q - (bx - ax) * r) / (2. * (abs(max([abs(q - r), tiny])) * sign(q - r)));
	ulim    = bx + glimit * (cx - bx);
       
    if (((bx - u) * (u - cx)) > 0.)
        
        % --- Enters this branch if u is in between bx and cx, that is, cx < u < bx or bx < u < cx 
        fu = f1dim(u, x, p, costfunctional);
        
        if (fu < fc)
            % --- Enters here if u is in between bx and cx and fu < fc < fb
            % (the last inequality is due to the while loop) => we have a
            % minimum between bx and cx
            ax = bx;
            fa = fb;
            bx = u;
            fb = fu;
            return;
        else
            if (fu > fb)
                % --- Enters here if fb < fu and fb < fa (the last
                % inequality is due to the first swap) => we have a minimum
                % between ax and u
                cx = u;
                fc = fu;
                return;
            end
        end
        
        % --- No minimum found yet. Use default magnification.
        u = cx + gold * (cx - bx); 
        fu=f1dim(u, x, p, costfunctional);
   
    else
        
        if ((cx - u) * (u - ulim) > 0.)
        
            % --- Enters this branch if u is in between cx and its allowed limit ulim         
            fu=f1dim(u, x, p, costfunctional);
        
            if (fu < fc)
                % --- fa > fb > fc > fu => the function is decreasing
                % towards fu => shift everything towards u
                bx = cx;
                cx = u;
                u  = cx + gold * (cx - bx);
                fb = fc;
                fc = fu;
                fu = f1dim(u, x, p, costfunctional);
            end
        
        else
            if ((u - ulim) * (ulim - cx) >= 0.)
                % --- ulim is between u and cx which means that u is beyond
                % its maximum allowed value => limit u to its maximum value
                % ulim
                u  = ulim;
                fu = f1dim(u, x, p, costfunctional);
            else
                % --- Move u ahead with default magnification
                u  = cx + gold * (cx - bx);
                fu = f1dim(u, x, p, costfunctional);
            end
        end
    end
    % --- Update the points
    ax = bx;
    bx = cx;
    cx = u;
    fa = fb;
    fb = fc;
    fc = fu;

end


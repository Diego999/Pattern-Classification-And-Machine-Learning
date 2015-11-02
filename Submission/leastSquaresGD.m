% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [beta] = leastSquaresGD(y, tX, alpha)
    % algorithm parameters
    maxIters = 35000;
    epsilon_convergence = 1e-8;

    % initialize beta
    D = size(tX,2)-1;
    beta = zeros(D+1,1);
    %steps = maxIters;
    
    for k = 1:maxIters
        % Compute gradient
        g = computeGradient(y, tX, beta);
        
        % Cost function
        % L(k) = computeCost(y, tX, beta);
        
        % Update beta
        beta = beta - alpha*g;
        
        % Check convergence
        if(norm(g) < epsilon_convergence)
            %steps = k;
            break;
        end
    end
end

function [gradient] = computeGradient(y, tX, beta)
    N = length(y);
    e = y - tX*beta;
    gradient = -(1.0/N)*tX'*e;
end

function [e] = computeCost(y, tX, beta)
    N = length(y);
    e = y - tX*beta;
    e = e'*e/(2.0*N);
end

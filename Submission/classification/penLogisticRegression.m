% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [beta] = penLogisticRegression(y, tX, alpha, lambda)
    % algorithm parameters
    maxIters = 100000;
    epsilon_convergence = 1e-6;

    % initialize beta
    D = size(tX,2)-1;
    beta = zeros(D+1,1);
    %steps = maxIters;
    
    for k = 1:maxIters
        % Compute gradient
        g = computeGradient(y, tX, beta, lambda);
        
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

function [gradient] = computeGradient(y, tX, beta, lambda)
    penalized_term = 2*lambda*beta;
    
    % We don't penalized beta zero
    penalized_term(1,1) = 0;
    
    gradient = tX'*(sigmoid(tX*beta) - y) + penalized_term;
end

function [res] = sigmoid(x)
    ex = exp(x);
    res = ex./(1 + ex);
end

function [res] = computeCost(y, tX, beta)
    tXBeta = tX*beta;
    res = -sum(y.*tXBeta - log(1 + exp(tXBeta)));
end

% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [beta] = logisticRegression(y, tX, alpha)
    % algorithm parameters
    maxIters = 1000000;
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
    fprintf('%.10f\n', norm(g));
end

function [gradient] = computeGradient(y, tX, beta)
    gradient = tX'*(sigmoid(tX*beta) - y);
end

function [res] = computeCost(y, tX, beta)
    tXBeta = tX*beta;
    res = -sum(y.*tXBeta - log(1 + exp(tXBeta)));
end

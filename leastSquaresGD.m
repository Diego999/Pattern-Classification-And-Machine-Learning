% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [beta] = leastSquaresGD(y, tX, alpha)
    % algorithm parameters
    maxIters = 100000;
    epsilon_convergence = 1e-6;

    % initialize beta
    D = size(tX,2);
    beta = randn(D,1);
    steps = 0;
    
    for k = 1:maxIters
        % Compute gradient
        g = computeGradient(y, tX, beta);
        
        %Update beta
        beta = beta - alpha*g;

        % Check convergence
        if(g'*g < epsilon_convergence)
            steps = k;
            break;
        end
    end
    
    % Some information
    %L = computeCost(y, tX, beta);
    %fprintf('%d\n', steps);
    %fprintf('K : %d \n Beta : %s\n Compute Cost : %.4f', steps, sprintf('%.4f ', beta), L);
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

% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [e] = logLoss(y, p)
    N = length(y);
    res = 0;
    
    for i = 1:1:N
       res = res + y(i)*p(i) - log(1.0 + exp(p(i)));
    end
    
    e = -res/N;
end


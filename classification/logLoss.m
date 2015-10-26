% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [e] = logLoss(y, p)
    N = length(y);
    res = 0;
    
    for i = 1:1:N
       res = res + y(i)*log(p(i)) + (1-y(i))*log(1-p(i));
    end
    
    e = -res/N;
end


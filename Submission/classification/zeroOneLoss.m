% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [e] = zeroOneLoss(y, yhat)
    N = length(y);
    res = 0;
    
    for i = 1:1:N
       if y(i) ~= yhat(i)
           res = res + 1;
       end
    end
    
    e = res/N;
end


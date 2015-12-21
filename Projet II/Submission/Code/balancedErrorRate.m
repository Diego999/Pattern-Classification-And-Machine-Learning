% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [e] = balancedErrorRate(y, yhat)
    N = length(y);
    res = 0;
    
    classes = 1:1:(length(unique(y)));
    
    for c = 1:1:length(classes)
        res2 = 0;
        for i = 1:1:N
           if y(i) == classes(c) && y(i) ~= yhat(i)
               res2 = res2 + 1;
           end
        end
        res = res + res2/length(find(y == classes(c)));
    end
    e = res/length(classes);
end


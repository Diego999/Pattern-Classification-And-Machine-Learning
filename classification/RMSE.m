% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [e] = RMSE(y, p)
    N = length(y);
    e = sqrt(sum((y-p).^2)/N);
end


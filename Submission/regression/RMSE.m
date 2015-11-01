% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [e] = RMSE(y, yhat)
    N = length(y);
    e = sqrt(sum((y-yhat).^2)/N);
end


% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [res] = sigmoid(x)
    ex = exp(x);
    res = ex./(1 + ex);
end
% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [beta] = leastSquares(y, tX)
    beta = (tX'*tX) \ (tX'*y);
end


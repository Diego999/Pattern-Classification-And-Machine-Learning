% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [beta] = ridgeRegression(y, tX, lambda)
    tXTmultX = tX'*tX;
    
    % tXTmultX is a square matrix, so no problem with length
    I = eye(length(tXTmultX));   
    
    % We do not touch beta_0
    I(1,1) = 0;
    
    beta = (tX'*tX + lambda*I)\tX'*y;
end


% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [Te] = createTestingCNN(X)

    % Initialize Tr, Te
    Te = [];

    % CNN is a sparse matrix, better to use sparse representation
    Te.X = sparse(double(X));
    
    % Normalizing
    [Te.nX, u, s] = zscore(Te.X);
end

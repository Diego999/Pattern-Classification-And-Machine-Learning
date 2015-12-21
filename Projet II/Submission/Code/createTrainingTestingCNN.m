% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [Tr, Te] = createTrainingTestingCNN(X, y, ratio)

    % Initialize Tr, Te
    Tr = [];
    Te = [];

    % Permute IDs and compute middle
    idx = randperm(size(X,1));
    mid = floor(length(idx) * ratio);

    % CNN is a sparse matrix, better to use sparse representation
    Xsparse = sparse(double(X));

    % preparing training
    Tr.idxs = idx(1:mid);
    Tr.X = Xsparse(Tr.idxs,:);
    Tr.y = y(Tr.idxs);

    % preparing testing
    Te.idxs = idx(mid+1:end);
    Te.X = Xsparse(Te.idxs,:);
    Te.y = y(Te.idxs);

    % Normalizing
    [Tr.nX, u, s] = zscore(Tr.X);
    Te.nX = normalize(Te.X, u, s);

end

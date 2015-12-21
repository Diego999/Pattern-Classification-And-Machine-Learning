% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [Tr, Te] = createTrainingTestingHOG(X, y, ratio)

    % Initialize Tr, Te
    Tr = [];
    Te = [];

    % Permute IDs and compute middle
    idx = randperm(size(X,1));
    mid = floor(length(idx) * ratio);

    % preparing training
    Tr.idxs = idx(1:mid);
    Tr.X = X(Tr.idxs,:);
    Tr.y = y(Tr.idxs);

    % preparing testing
    Te.idxs = idx(mid+1:end);
    Te.X = X(Te.idxs,:);
    Te.y = y(Te.idxs);

    % Normalizing
    [Tr.nX, u, s] = zscore(Tr.X);
    Te.nX = normalize(Te.X, u, s);

end

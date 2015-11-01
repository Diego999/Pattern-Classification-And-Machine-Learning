% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [XNorm] = normalizedData(X)
    XNorm = X-repmat(mean(X), size(X, 1), 1);
    XNorm = XNorm./repmat(std(XNorm), size(X, 1), 1);
end


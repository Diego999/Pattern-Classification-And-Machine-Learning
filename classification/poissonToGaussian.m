% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [X] = poissonToGaussian(X, features)
    for f = 1:1:length(features)
        feature = features(f);
        
        % Using Abs
        X(:,feature) = abs(X(:,feature)).^0.25;
        
        % Holding negative values
        %neg_idx = find(X(:,feature) < 0);
        %X(neg_idx, feature) = -X(neg_idx, feature);
        %X(:,feature) = X(:,feature).^0.25;
        %X(neg_idx, feature) = -X(neg_idx, feature);
    end
end
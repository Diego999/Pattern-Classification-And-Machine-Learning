% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3] = preprocess(y, X)
    [idx_cls1, idx_cls2, idx_cls3] = findClusters(y, X);
    
    y_cls1 = y(idx_cls1);
    X_cls1 = X(idx_cls1);
    
    y_cls2 = y(idx_cls2);
    X_cls2 = X(idx_cls2);
    
    y_cls3 = y(idx_cls3);
    X_cls3 = X(idx_cls3);
end

function [idx_cls1, idx_cls2, idx_cls3] = findClusters(y, X)
    yN = normalizedData(y);

    % Find first separation of cluster using feature 2
    idx_cluster1 = find(yN < 0);
    idx_others_clusters = setdiff(1:length(y), idx_cluster1)';

    % There are 4 outliers here !
    outliers_cluster1 = find(yN > 0 & yN < 0.1 | yN > 3.5);

    % Find second separation of cluster using feature 16
    idx_cluster2 = find(yN < 1.65);
    idx_cluster3 = setdiff(idx_others_clusters, idx_cluster2);

    % There are 2 outliers here ! (there is 2 times the number 653)
    outliers_cluster3 = find(yN < 1.54 & yN > 1.53 | yN > 3.5);

    idx_cluster2 = setdiff(idx_others_clusters, idx_cluster3);

    % Some checks
    assert(length(idx_cluster1) + length(idx_cluster2) + length(idx_cluster3) == length(X));
    
    % Write cluster idx without outliers
    idx_cls1 = setdiff(setdiff(idx_cluster1, outliers_cluster1), outliers_cluster3);
    idx_cls2 = setdiff(setdiff(idx_cluster2, outliers_cluster1), outliers_cluster3);
    idx_cls3 = setdiff(setdiff(idx_cluster3, outliers_cluster1), outliers_cluster3);
end
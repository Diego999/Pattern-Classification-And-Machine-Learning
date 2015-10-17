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
    XN = normalizedData(X);    
    yN = normalizedData(y);

    % Find first separation of cluster using feature 2
    idx_cluster1 = find(XN(:,2) < 0.42);
    idx_others_clusters = setdiff(1:length(y), idx_cluster1)';
    
    % There are 12 outliers here !
    outliers_cluster1 = find(yN > 3.5 | (XN(:,2) < 0.42 & yN > 0.4) | (XN(:,2) > 0.42 & yN < 0));

    % Find second separation of cluster using feature 16
    idx_cluster3 = setdiff(idx_others_clusters, find(yN < 1.65 | XN(:,16) < 0.7));

    % There is 11 outlier here
    outliers_cluster3 = find(yN > 2 & XN(:,16) < 1.17);

    % Find the last cluster using the last points
    idx_cluster2 = setdiff(idx_others_clusters, idx_cluster3);

    % There are 5 outliers here !
    outliers_cluster2 = find(yN > 3.5 | (XN(:,16) > 1.17 & yN > 0.5 & yN < 1.5));

    % Some checks
    assert(length(idx_cluster1) + length(idx_cluster2) + length(idx_cluster3) == length(X));
    
    % Write cluster idx without outliers
    idx_cls1 = setdiff(setdiff(setdiff(idx_cluster1, outliers_cluster1), outliers_cluster2), outliers_cluster3);
    idx_cls2 = setdiff(setdiff(setdiff(idx_cluster2, outliers_cluster1), outliers_cluster2), outliers_cluster3);
    idx_cls3 = setdiff(setdiff(setdiff(idx_cluster3, outliers_cluster1), outliers_cluster2), outliers_cluster3);
end
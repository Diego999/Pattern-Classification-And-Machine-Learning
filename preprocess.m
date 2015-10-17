% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(y, X)
    [idx_cls1, idx_cls2, idx_cls3] = findClusters(y, X);
    
    % Find the discrete features
    [idx_feature_2, idx_feature_3, idx_feature_4] = findDiscreteFeatures(X);
    idx_feature_real = setdiff(setdiff(setdiff(1:1:size(X,2), idx_feature_2), idx_feature_3), idx_feature_4);

    % We shouldn't normalize y
    % Normalize each cluster independantely withou categorical features
    y_cls1 = y(idx_cls1);
    X_cls1 = normalizedData(X(idx_cls1,idx_feature_real));
    
    y_cls2 = y(idx_cls2);
    X_cls2 = normalizedData(X(idx_cls2,idx_feature_real));
    
    y_cls3 = y(idx_cls3);
    X_cls3 = normalizedData(X(idx_cls3,idx_feature_real));
    
    % We encode categorical features and add the to the normalized X
    X_cls1 = dummyFeatureEncoding(X_cls1, X(idx_cls1,:), idx_feature_2, idx_feature_3, idx_feature_4);
    X_cls2 = dummyFeatureEncoding(X_cls2, X(idx_cls2,:), idx_feature_2, idx_feature_3, idx_feature_4);
    X_cls3 = dummyFeatureEncoding(X_cls3, X(idx_cls3,:), idx_feature_2, idx_feature_3, idx_feature_4);
end

function [XX] = dummyFeatureEncoding(X,X_original, idx_feature_2, idx_feature_3, idx_feature_4)
    XX = zeros(length(X_original),1);
    % The index to copy the real values (non discrete) of X
    i_real_X = 1;
    for i = 1:1:size(X_original,2)
        % If categorical of 2, just copy the cols
        if any(idx_feature_2 == i)
          XX(:,end+1) = X_original(:,i);
        % If categorical of 3 or 4, create (k-1) cols and put 0-0-0 if 0,
        % 1-0-0 if 1, 0-1-0 if 2, 0-0-1 if 4 etc.
        elseif any(idx_feature_3 == i)
            for j = 1:1:(3-1)
                cols = zeros(length(X),1);
                cols(find(X_original(:,i) == j),1) = 1;
                XX(:,end+1) = cols;
            end
        elseif any(idx_feature_4 == i)
            for j = 1:1:(4-1)
                cols = zeros(length(X),1);
                cols(find(X_original(:,i) == j),1) = 1;
                XX(:,end+1) = cols;
            end
        % Copy the real values (non discrete)
        else
            XX(:,end+1) = X(:,i_real_X);
            i_real_X = i_real_X + 1;
        end
    end
    
    % Remove the 1st column
    XX = XX(:,2:end);
end

function [idx_feature_2, idx_feature_3, idx_feature_4] = findDiscreteFeatures(X)
    idx_feature_2 = [];
    idx_feature_3 = [];
    idx_feature_4 = [];
    
    for i = 1:1:size(X,2)
       unique_val = length(unique(X(:,i)));
       if unique_val == 2
        idx_feature_2 = [idx_feature_2 i];
       elseif unique_val == 3
        idx_feature_3 = [idx_feature_3 i];
       elseif unique_val == 4
        idx_feature_4 = [idx_feature_4 i];
       end
    end
    
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
% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

% If degrees are not specified, they will get the value 1.
function [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(y, X, degree1, degree2, degree3)
    if ~exist('degree1', 'var')
        degree1 = 1;
    end
    
    if ~exist('degree2', 'var')
        degree2 = 1;
    end
    
    if ~exist('degree3', 'var')
        degree3 = 1;
    end
    
    % Find idx of each cluster
    [idx_cls1, idx_cls2, idx_cls3] = findClusters(y, X);
    
    % Find different type of features
    [idx_feature_2, idx_feature_3, idx_feature_4] = findDiscreteFeatures(X);
    idx_feature_real = setdiff(setdiff(setdiff(1:1:size(X,2), idx_feature_2), idx_feature_3), idx_feature_4);

    % Preprocess per cluster
    [y_cls1, X_cls1] = preprocessCluster(y, X, idx_cls1, degree1, idx_feature_2, idx_feature_3, idx_feature_4, idx_feature_real);
    [y_cls2, X_cls2] = preprocessCluster(y, X, idx_cls2, degree2, idx_feature_2, idx_feature_3, idx_feature_4, idx_feature_real);
    [y_cls3, X_cls3] = preprocessCluster(y, X, idx_cls3, degree3, idx_feature_2, idx_feature_3, idx_feature_4, idx_feature_real);
end

% idx is the idx of the current cluster, idx_feature_3 are the idx of the
% discrete features, and idx_features_real the real ones.
function [y, X] = preprocessCluster(y_, X_, idx, degree, idx_feature_2, idx_feature_3, idx_feature_4, idx_feature_real)

    % If we decide to remove some features
    %[res_corr, idx_corr] = sort(abs(corr(X_(idx,:),y_(idx,:))));
    X__ = X_;
    %X__(:,idx_corr(1:5)) = [];
    X = X__;
        
    % Build the poly for X
    X = repmat(X(idx,:),1,degree);
    nbFeatures = size(X_,2);

    % Prepare to extend idx_feature_i for the polynomial features
    idx_feature_2_poly = idx_feature_2;
    idx_feature_3_poly = idx_feature_3;
    idx_feature_4_poly = idx_feature_4;
    
    real_features_idx_to_normalize = idx_feature_real;
    for d = 1:1:(degree-1)
        % Limit of the i'th degree
        beg = d*nbFeatures+1;
        en = (d+1)*nbFeatures;
        
        % Compute the new idx of features of the current degree
        real_features_idx = idx_feature_real + (size(X_,2)*d);
        idx_feature_2_this_degree = idx_feature_2 + (size(X_,2)*d);
        idx_feature_3_this_degree = idx_feature_3 + (size(X_,2)*d);
        idx_feature_4_this_degree = idx_feature_4 + (size(X_,2)*d);
        
        % Do the union between old idx and the new one
        real_features_idx_to_normalize = union(real_features_idx_to_normalize, real_features_idx);
        idx_feature_2_poly = union(idx_feature_2_poly, idx_feature_2_this_degree);
        idx_feature_3_poly = union(idx_feature_3_poly, idx_feature_3_this_degree);
        idx_feature_4_poly = union(idx_feature_4_poly, idx_feature_4_this_degree);
    
        % Update only the real features, because we don't raise discrete
        % variable to power
        cols = intersect(real_features_idx, beg:1:en);
        X(:,cols) = X(:,cols).^(d+1);
    end
    
    X__ = X;
    
    % Normalize
    y = y_(idx);
    % WITH NORMALIZATION
    X = normalizedData(X(:,real_features_idx_to_normalize));
    % WITHOUT NORMALIZATION
    % X = X(:,real_features_idx_to_normalize);
    
    % WITH DUMMY ENCODING
    % We encode categorical features and add them to the normalized X
    X = dummyFeatureEncoding(X, X__, idx_feature_2_poly, idx_feature_3_poly, idx_feature_4_poly); 
    % WITHOUT DUMMY ENCODING
    % X = addCategoricalVariableWithoutEncoding(X, X__, idx_feature_2_poly, idx_feature_3_poly, idx_feature_4_poly);

end

% X is the matrix with only the real features
% X_original is the original matrix (real + discrete features)
function [XX] = dummyFeatureEncoding(X, X_original, idx_feature_2, idx_feature_3, idx_feature_4)
        XX = zeros(size(X_original, 1),1);
    
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
                cols = zeros(size(X, 1),1);
                cols(find(X_original(:,i) == j),1) = 1;
                XX(:,end+1) = cols;
            end
            
        % Categorical of 4
        elseif any(idx_feature_4 == i)
            for j = 1:1:(4-1)
                cols = zeros(size(X,1),1);
                cols(find(X_original(:,i) == j),1) = 1;
                XX(:,end+1) = cols;
            end
            
        % Copy the real values (non discrete)
        else
            XX(:,end+1) = X(:,i_real_X);
            i_real_X = i_real_X + 1;
        end
    end
    
    % Remove the 1st column (full of 0)
    XX = XX(:,2:end);
end

function [XX] = addCategoricalVariableWithoutEncoding(X,X_original, idx_feature_2, idx_feature_3, idx_feature_4)
    XX = zeros(size(X_original,1),1);
    % The index to copy the real values (non discrete) of X
    i_real_X = 1;
    for i = 1:1:size(X_original,2)
        % If categorical of 2, just copy the cols
        if any(idx_feature_2 == i) || any(idx_feature_3 == i) || any(idx_feature_4 == i)
          XX(:,end+1) = X_original(:,i);
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

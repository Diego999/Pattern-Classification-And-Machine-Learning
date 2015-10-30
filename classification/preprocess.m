% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

% If degrees are not specified, they will get the value 1.
function [y, X, outliers_idx] = preprocess(y_, X_, degree)
     if ~exist('degree', 'var')
            degree = 1;
     end
    
    % Find different type of features
    [idx_feature_2, idx_feature_4, idx_feature_5] = findDiscreteFeatures(X_);
    idx_feature_real = setdiff(setdiff(setdiff(1:1:size(X_,2), idx_feature_2), idx_feature_4), idx_feature_5);

    X = X_;
    
    %X = poissonToGaussian(X, [1 5 8 11 12 13 20 23 24 7 16 17 22 9 10 14 21 6 19 2]);
    X = poissonToGaussian(X, idx_feature_real);
    
    % 2
    y = y_;
    
    % Transform -1 to 0
    y(find(y == -1)) = 0;
    
    %[y, X, outliers_idx] = removeOutliers(y, X_);
    %outliers_idx = [];
    
    % Build the poly for X
    X = repmat(X,1,degree);
    nbFeatures = size(X_,2);

    % Prepare to extend idx_feature_i for the polynomial features
     idx_feature_2_poly = idx_feature_2;
     idx_feature_4_poly = idx_feature_4;
     idx_feature_5_poly = idx_feature_5;
     
     real_features_idx_to_normalize = idx_feature_real;
    for d = 1:1:(degree-1)
        % Limit of the i'th degree
        beg = d*nbFeatures+1;
        en = (d+1)*nbFeatures;
        
        % Compute the new idx of features of the current degree
        real_features_idx = idx_feature_real + (size(X_,2)*d);
        idx_feature_2_this_degree = idx_feature_2 + (size(X_,2)*d);
        idx_feature_4_this_degree = idx_feature_4 + (size(X_,2)*d);
        idx_feature_5_this_degree = idx_feature_5 + (size(X_,2)*d);
        
        % Do the union between old idx and the new one
        real_features_idx_to_normalize = union(real_features_idx_to_normalize, real_features_idx);
        idx_feature_2_poly = union(idx_feature_2_poly, idx_feature_2_this_degree);
        idx_feature_4_poly = union(idx_feature_4_poly, idx_feature_4_this_degree);
        idx_feature_5_poly = union(idx_feature_5_poly, idx_feature_5_this_degree);
    
        % Update only the real features, because we don't raise discrete
        % variable to power
        cols = intersect(real_features_idx, beg:1:en);
        X(:,cols) = X(:,cols).^(d+1);
    end
    
    X_ = X;
    % Normalize real values
    X = normalizedData(X(:,real_features_idx_to_normalize));
    
    % WITH DUMMY ENCODING
    % We encode categorical features and add them to the normalized X
    X = dummyFeatureEncoding(X, X_, idx_feature_2_poly, idx_feature_4_poly, idx_feature_5_poly); 
    % WITHOUT DUMMY ENCODING
    % X = addCategoricalVariableWithoutEncoding(X, X_, idx_feature_2_poly, idx_feature_4_poly, idx_feature_5_poly);
end

function [y, X, outliers_idx] = removeOutliers(y_, X_)
    outliers_idx = [];
    X__ = normalizedData(X_);
    features = [6 14 19 20 21 23 24];
    for f = 1:1:length(features)
       outliers_idx = union(outliers_idx, find(X__(:, features(f)) >= 4.5));
       outliers_idx = union(outliers_idx, find(X__(:, features(f)) <= -4.5));
    end
    
    outliers_idx = [outliers_idx; find(X__(:, 22) <= -2.5)];
    
    outliers_idx = [outliers_idx; find(X__(:, 17) >= 4.0)];
    outliers_idx = [outliers_idx; find(X__(:, 17) <= -4.0)];
       
    outliers_idx = [outliers_idx; find(X__(:, 10) >= 4.5)];
    outliers_idx = [outliers_idx; find(X__(:, 10) <= -2.0)];
    outliers_idx = [outliers_idx; find(X__(:, 16) >= 4.5)];
    outliers_idx = [outliers_idx; find(X__(:, 16) <= -2.0)];   
    
    outliers_idx = [outliers_idx; find(X__(:, 13) >= 2)];
    outliers_idx = [outliers_idx; find(X__(:, 13) <= -4.5)];
    
    outliers_idx = [outliers_idx; find(X__(:, 11) >= 1.5)];
    outliers_idx = [outliers_idx; find(X__(:, 11) <= -4.0)];
    outliers_idx = [outliers_idx; find(X__(:, 12) >= 1.5)];
    outliers_idx = [outliers_idx; find(X__(:, 12) <= -4.0)];
    
    outliers_idx = [outliers_idx; find(X__(:, 9) <= -4.0)];
    
    outliers_idx = [outliers_idx; find(X__(:, 8) >= 2.0)];
    outliers_idx = [outliers_idx; find(X__(:, 8) <= -4.0)];
    
    outliers_idx = [outliers_idx; find(X__(:, 7) >= 4.5)];
    outliers_idx = [outliers_idx; find(X__(:, 7) <= -1.0)];
    
    outliers_idx = [outliers_idx; find(X__(:, 5) >= 1.5)];
    outliers_idx = [outliers_idx; find(X__(:, 5) <= -4.5)];
    
    outliers_idx = [outliers_idx; find(X__(:, 1) >= 1.5)];
    outliers_idx = [outliers_idx; find(X__(:, 1) <= -4.5)];
    
    outliers_idx = [outliers_idx; find(X__(:, 2) <= -1.5)];
    
    [res, unique_id, temp] = unique(outliers_idx);
    outliers_idx = setdiff(1:1:length(outliers_idx), unique_id);
    idx = setdiff(1:1:length(y_), outliers_idx);
    y = y_(idx,:);
    X = X_(idx,:);
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
    % X = dummyFeatureEncoding(X, X__, idx_feature_2_poly, idx_feature_3_poly, idx_feature_4_poly); 
    % WITHOUT DUMMY ENCODING
    X = addCategoricalVariableWithoutEncoding(X, X__, idx_feature_2_poly, idx_feature_3_poly, idx_feature_4_poly);

end

% X is the matrix with only the real features
% X_original is the original matrix (real + discrete features)
function [XX] = dummyFeatureEncoding(X, X_original, idx_feature_2, idx_feature_4, idx_feature_5)
        XX = zeros(size(X_original, 1),1);
    
    % The index to copy the real values (non discrete) of X
    i_real_X = 1;
    for i = 1:1:size(X_original,2)
        
        % If categorical of 2, just copy the cols
        if any(idx_feature_2 == i)
          XX(:,end+1) = X_original(:,i);
        
          % If categorical of 3 or 4, create (k-1) cols and put 0-0-0 if 0,
        % 1-0-0 if 1, 0-1-0 if 2, 0-0-1 if 4 etc.
        elseif any(idx_feature_4 == i)
            for j = 1:1:(4-1)
                cols = zeros(size(X, 1),1);
                cols(find(X_original(:,i) == j),1) = 1;
                XX(:,end+1) = cols;
            end
            
        % Categorical of 4
        elseif any(idx_feature_5 == i)
            for j = 1:1:(5-1)
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

function [XX] = addCategoricalVariableWithoutEncoding(X,X_original, idx_feature_2, idx_feature_4, idx_feature_5)
    XX = zeros(size(X_original,1),1);
    % The index to copy the real values (non discrete) of X
    i_real_X = 1;
    for i = 1:1:size(X_original,2)
        % If categorical of 2, just copy the cols
        if any(idx_feature_2 == i) || any(idx_feature_4 == i) || any(idx_feature_5 == i)
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

function [idx_feature_2, idx_feature_4, idx_feature_5] = findDiscreteFeatures(X)
    idx_feature_2 = [];
    idx_feature_4 = [];
    idx_feature_5 = [];
    
    for i = 1:1:size(X,2)
       unique_val = length(unique(X(:,i)));
       if unique_val == 2
        idx_feature_2 = [idx_feature_2 i];
       elseif unique_val == 4
        idx_feature_4 = [idx_feature_4 i];
       elseif unique_val == 5
        idx_feature_5 = [idx_feature_5 i];
       end
    end
    
end

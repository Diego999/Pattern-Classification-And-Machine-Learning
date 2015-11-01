% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

% Add in preprocessCluster around line 68 the normalization 
% Remove in preprocessCluster around line 75 the dummy encoding

clear all;
close all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;
N = length(y);

k1 = 10;
k2 = 8;
k3 = 6;

d1 = 3;
d2 = 6;
d3 = 3;

lambda_cls1 = 0.0369;
lambda_cls2 = 1.8303e-4;
lambda_cls3 = 0.0053;
    
%% COMPUTING ESTIMATE ERROR
K = 10;
N = length(y);
idxCV = splitGetCV(K, N);
setSeed(28111993);
shuffle = randperm(N);
XRandom = X(shuffle,:);
yRandom = y(shuffle,:);

for k = 1:1:K;
    [XTr, yTr, XTe, yTe] = splitGetTrTe(yRandom, XRandom, idxCV, k);

    [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTr, XTr, d1, d2, d3);

    tX_cls1 = [ones(length(y_cls1),1) X_cls1];
    tX_cls2 = [ones(length(y_cls2),1) X_cls2];
    tX_cls3 = [ones(length(y_cls3),1) X_cls3];

    beta_cls1 = ridgeRegression(y_cls1, tX_cls1, lambda_cls1);
    beta_cls2 = ridgeRegression(y_cls2, tX_cls2, lambda_cls2);
    beta_cls3 = ridgeRegression(y_cls3, tX_cls3, lambda_cls3);

    [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTe, XTe, d1, d2, d3, 1);

    tXTe_cls1 = [ones(length(y_cls1),1) X_cls1];
    tXTe_cls2 = [ones(length(y_cls2),1) X_cls2];
    tXTe_cls3 = [ones(length(y_cls3),1) X_cls3];

    N_ = length(y_cls1)+length(y_cls2)+length(y_cls3);
    err(k) = sqrt((1/N_) * (sum((y_cls1 - tXTe_cls1*beta_cls1).^2) ...
                         +  sum((y_cls2 - tXTe_cls2*beta_cls2).^2) ...
                         +  sum((y_cls3 - tXTe_cls3*beta_cls3).^2)));
end

[y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(y, X);

 mean_err = mean(err);
 std_err = std(err);
 
%% TRAINING FOR PREDICTION

[y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(y, X, d1, d2, d3);

tX_cls1 = [ones(length(y_cls1),1) X_cls1];
tX_cls2 = [ones(length(y_cls2),1) X_cls2];
tX_cls3 = [ones(length(y_cls3),1) X_cls3];

% Compute betas
beta_cls1 = ridgeRegression(y_cls1, tX_cls1, lambda_cls1);
beta_cls2 = ridgeRegression(y_cls2, tX_cls2, lambda_cls2);
beta_cls3 = ridgeRegression(y_cls3, tX_cls3, lambda_cls3);

% Prepare the data to be predicted
X = X_test;
y = zeros(size(X, 1), 1);

% y is not used in preprocess
[y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(y, X, d1, d2, d3, 1);

tX_cls1 = [ones(length(y_cls1),1) X_cls1];
tX_cls2 = [ones(length(y_cls2),1) X_cls2];
tX_cls3 = [ones(length(y_cls3),1) X_cls3];

y_cls1 = tX_cls1*beta_cls1;
y_cls2 = tX_cls2*beta_cls2;
y_cls3 = tX_cls3*beta_cls3;

% Writing back in y

for n = 1:1:size(X,1)
    idx = find(idx_cls1 == n);
    if ~isempty(idx)
        y(n) = y_cls1(idx);
    else
        idx = find(idx_cls2 == n);
        if ~isempty(idx)
            y(n) = y_cls2(idx);
        else
            idx = find(idx_cls3 == n);
            y(n) = y_cls3(idx);
        end
    end
end

csvwrite('predictions_regression.csv', y);
% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

%clear all;
%close all;

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
    
    lambda_cls1 = 0.0369;
    lambda_cls2 = 1.8303e-4;
    lambda_cls3 = 0.0053;

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

    err_cls1(k) = RMSE(y_cls1, tXTe_cls1*beta_cls1);
    err_cls2(k) = RMSE(y_cls2, tXTe_cls2*beta_cls2);
    err_cls3(k) = RMSE(y_cls3, tXTe_cls3*beta_cls3);
end

[y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(y, X);
err = sqrt((1/N)*( ...
            (size(X_cls1, 1)*(err_cls1.^2)) ...
          + (size(X_cls2, 1)*(err_cls2.^2)) ...
          + (size(X_cls3, 1)*(err_cls3.^2)) ...
     ));
 mean_err = mean(err);
 std_err = std(err);
 
%% TRAINING FOR PREDICTION
lambda_cls1 = 0.0369;
lambda_cls2 = 1.8303e-4;
lambda_cls3 = 0.0053;

[y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(y, X, d1, d2, d3);

tX_cls1 = [ones(length(y_cls1),1) X_cls1];
tX_cls2 = [ones(length(y_cls2),1) X_cls2];
tX_cls3 = [ones(length(y_cls3),1) X_cls3];

beta_cls1 = ridgeRegression(y_cls1, tX_cls1, lambda_cls1);
beta_cls2 = ridgeRegression(y_cls2, tX_cls2, lambda_cls2);
beta_cls3 = ridgeRegression(y_cls3, tX_cls3, lambda_cls3);

% PREPARE THE DATA TO PREDICT
X = X_test;
y = zeros(size(X, 1), 1);

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
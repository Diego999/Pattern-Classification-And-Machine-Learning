% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;
close all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;
N = length(y);
[y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3] = preprocess(y, X);

% Methodology
% LS, won't work well due to ill-conditioning
% LSGD
% Ridge

% 1) Without any preprocess (no dummy encoding, no cluster separation, no normalization)
% 2) Same as 1) but with normalization
% 3) Separation with the cluster and normalization per cluster
% 4) Same as 3 but with dummy encoding
% 5) Same as 3 but with Polynomial basis
% 6) Same as 4 but with Polynomial basis
% 6) Remove some uncorrelated data

% **********************************
%             TEST 1
% **********************************
% Mean Tr : 1219
% Mean Te : 1219
% K = 7;
% y_ = y;
% X_ = X;
% alpha = 0.002875;
% lambda = 0.15;%findLambda(K, y_, X_);
% [mean_err1_tr_ls, mean_err1_te_ls, mean_err1_tr_lsgd, mean_err1_te_lsgd, mean_err1_tr_rr, mean_err1_te_rr] = runTest(K, y_, X_, alpha, lambda);

% **********************************
%             TEST 2
% **********************************
% Mean Tr : 1219
% Mean Te : 1219
% K = 7;
% y_ = y;
% X_ = normalizedData(X);
% alpha = 0.002875;
% lambda = 0.15;%findLambda(K, y_, X_);
% [mean_err2_tr_ls, mean_err2_te_ls, mean_err2_tr_lsgd, mean_err2_te_lsgd, mean_err2_tr_rr, mean_err2_te_rr] = runTest(K, y_, X_, alpha, lambda);

% **********************************
%             TEST 3
% **********************************
% WARNING : Change preprocess fct to remove dummy encoding !
% K = 5 2 2
% Mean Tr : 121 + 141 + 151 = 413
% Mean Te : 127 + 198 + 338 = 663

% K = 5 3 3
% Mean Tr : 121 + 150 + 175 = 446
% Mean Te : 127 + 182 + 276 = 585

% Cluster 1
% Mean Tr : 121
% Mean Te : 127
% K = 5;
% y_ = y_cls1;
% X_ = X_cls1;
% alpha = 0.01;
% lambda = 0.15;%findLambda(K, y_, X_);
% [mean_err3_tr_ls_cls1, mean_err3_te_ls_cls1, mean_err3_tr_lsgd_cls1, mean_err3_te_lsgd_cls1, mean_err3_tr_rr_cls1, mean_err3_te_rr_cls1] = runTest(K, y_, X_, alpha, lambda);
% 
% % Cluster 2
% % Mean Tr : 149 (or 150 with K = 3)
% % Mean Te : 187 (or 182 with K = 3)
% K = 3;
% y_ = y_cls2;
% X_ = X_cls2;
% alpha = 0.01;
% lambda = 0.15;%findLambda(K, y_, X_);
% [mean_err3_tr_ls_cls2, mean_err3_te_ls_cls2, mean_err3_tr_lsgd_cls2, mean_err3_te_lsgd_cls2, mean_err3_tr_rr_cls2, mean_err3_te_rr_cls2] = runTest(K, y_, X_, alpha, lambda);
% 
% % Cluster 3
% % Mean Tr 165 (or 175 with K = 3)
% % Mean Te : 313 (or 276 with K = 3)
% K = 3;
% y_ = y_cls3;
% X_ = X_cls3;
% alpha = 0.01;
% lambda = 0.15;%findLambda(K, y_, X_);
% [mean_err3_tr_ls_cls3, mean_err3_te_ls_cls3, mean_err3_tr_lsgd_cls3, mean_err3_te_lsgd_cls3, mean_err3_tr_rr_cls3, mean_err3_te_rr_cls3] = runTest(K, y_, X_, alpha, lambda);


% **********************************
%             TEST 4
% **********************************

% K = 5 2 2
% Mean Tr : 120 + 136 + 137 = 393 % Better
% Mean Te : 127 + 206 + 368 = 701 % Worse

% CAREFUL, D > N PROBLEM FOR y_cls3 !
% K = 5 3 2
% Mean Tr : 120 + 146 + 137 = 403 % A bit better
% Mean Te : 127 + 188 + 368 = 615 % Worse

% Cluster 1
% Mean Tr : 120
% Mean Te : 127
% K = 5;
% y_ = y_cls1;
% X_ = X_cls1;
% alpha = 0.07;
% lambda = 0.15;%findLambda(K, y_, X_);
% [mean_err4_tr_ls_cls1, mean_err4_te_ls_cls1, mean_err4_tr_lsgd_cls1, mean_err4_te_lsgd_cls1, mean_err4_tr_rr_cls1, mean_err4_te_rr_cls1] = runTest(K, y_, X_, alpha, lambda);
% 
% % Cluster 2
% % Mean Tr : 136 (or 146 with K = 3)
% % Mean Te : 206 (or 188 with K = 3)
% K = 3;
% y_ = y_cls2;
% X_ = X_cls2;
% alpha = 0.2;
% lambda = 0.15;%findLambda(K, y_, X_);
% [mean_err4_tr_ls_cls2, mean_err4_te_ls_cls2, mean_err4_tr_lsgd_cls2, mean_err4_te_lsgd_cls2, mean_err4_tr_rr_cls2, mean_err4_te_rr_cls2] = runTest(K, y_, X_, alpha, lambda);
% 
% % Cluster 3
% % Mean Tr : 137 (or 166 with K = 3)
% % Mean Te : 368 (or 300 with K = 3)
% K = 3;
% y_ = y_cls3;
% X_ = X_cls3;
% alpha = 0.2;
% lambda = 0.15;%findLambda(K, y_, X_);
% [mean_err4_tr_ls_cls3, mean_err4_te_ls_cls3, mean_err4_tr_lsgd_cls3, mean_err4_te_lsgd_cls3, mean_err4_tr_rr_cls3, mean_err4_te_rr_cls3] = runTest(K, y_, X_, alpha, lambda);

% **********************************
%             TEST 5
% **********************************
% WARNING : Change preprocess fct to remove dummy encoding !

% K = 5 5 3
% D = 3 5 2
% Mean Tr : 5.62 + 6.41 + 15.34 = 27.37
% Mean Te : 6.52 + 75.17 + 73.44 = 155.13

% Cluster 1
% K = 5;
% y_ = y_cls1;
% X_ = X_cls1;
% lambda = 0.15;%findLambda(K, y_, X_);
% [mean_err5_tr_cls1, mean_err5_te_cls1] = runTestPoly(K, y_, X_, lambda);

% Cluster 2
% K = 5;
% y_ = y_cls2;
% X_ = X_cls2;
% lambda = 0.15;%findLambda(K, y_, X_);
% [mean_err5_tr_cls2, mean_err5_te_cls2] = runTestPoly(K, y_, X_, lambda);
% 
% % Cluster 3
% K = 3;
% y_ = y_cls3;
% X_ = X_cls3;
% lambda = 0.15;%findLambda(K, y_, X_);
% [mean_err5_tr_cls3, mean_err5_te_cls3] = runTestPoly(K, y_, X_, lambda);

% **********************************
%             TEST 6
% **********************************
% WARNING : Don't forget to put the degree at 1 in preprocess
% K = 5 5 4
% D = 3 4 2
% Mean Tr : 5.63 + 15.9 + 17.45 = 38.98
% Mean Te : 6.52 + 77.67 + 64.14 = 148.33

%Cluster 1
% K = 5;
% lambda = 0.035;%findLambda(K, y, X, 1, 3);
% [mean_err5_tr_cls1, mean_err5_te_cls1] = runTestPoly(K, y, X, lambda, 1);

%Cluster 2
%K = 5;
%lambda = 0.2783;%findLambda(K, y, X, 2, 5);
% [mean_err5_tr_cls2, mean_err5_te_cls2] = runTestPoly(K, y, X, lambda, 2);
% 
% % Cluster 3
% K = 4;
% lambda = 0.7497;%findLambda(K, y, X, 3, 2);
% [mean_err5_tr_cls3, mean_err5_te_cls3] = runTestPoly(K, y, X, lambda, 3);
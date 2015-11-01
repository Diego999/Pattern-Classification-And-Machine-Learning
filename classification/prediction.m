% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

% We MUST normalize and transform -1 to 0 in order to have GD working
% Add categorical features from preprocess
% Add dummy encoding
% Add polynomial basis
% Add Transformation

clear all;
close all;

load('Sydney_classification.mat');

y = y_train;
X = X_train;
N = length(y);

K = 10;
degree = 2;
alpha = 0.001;

%% COMPUTING ESTIMATE ERROR

idxCV = splitGetCV(K, N);
setSeed(28111993);
shuffle = randperm(N);
XRandom = X(shuffle,:);
yRandom = y(shuffle,:);

for k = 1:1:K;
    [XTr, yTr, XTe, yTe] = splitGetTrTe(yRandom, XRandom, idxCV, k);

    [yTr, XTr] = preprocess(yTr, XTr, degree);   
    
    beta = logisticRegression(yTr, [ones(length(yTr), 1) XTr], alpha);
    
    [yTe, XTe] = preprocess(yTe, XTe, degree);
    
    tXTe = [ones(length(yTe),1) XTe];
    
    y_hat = (sigmoid(tXTe*beta) >= 0.5).*1.0;
    
    err(k) = RMSE(yTe, y_hat);
    err_01(k) = zeroOneLoss(yTe, y_hat);
    err_log(k) = logLoss(yTe, y_hat);
end

mean_err = mean(err);
std_err = std(err);
mean_err_01 = mean(err_01);
std_err_01 = std(err_01);
mean_err_log = mean(err_log);
std_err_log = std(err_log);

%% TRAINING FOR PREDICTION

[yTr, XTr] = preprocess(y, X, degree);   

beta = logisticRegression(yTr, [ones(length(yTr), 1) XTr], alpha);

% PREPARE THE DATA TO PREDICT
X = X_test;
y = zeros(size(X, 1), 1);

[y, X_] = preprocess(y, X, degree);
tXTe = [ones(length(y),1) X_];
y = (sigmoid(tXTe*beta) >= 0.5).*1.0;

csvwrite('predictions_classification.csv', y);
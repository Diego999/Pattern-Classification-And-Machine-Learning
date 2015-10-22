% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;
close all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;
N = length(y);

% 0) Constant baseline
% 0b) Constant baseline with clusters
% 1) Without any preprocess (no dummy encoding, no cluster separation, no normalization)
% 2) Same as 1) but with normalization
% 3) Separation with the cluster and normalization per cluster
% 4) Same as 3 but with dummy encoding
% 5) Same as 3 but with Polynomial basis
% 6) Same as 4 but with Polynomial basis

numberOfExperiments = 100;
proportionOfTraining = 0.8;

%%
% **********************************
%             TEST 0
% **********************************

for i = 1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
    
    % Nothing to learn
    beta = mean(yTr);
    
    err0(i) = RMSE(yTe, beta);
end

saveFile(err0, 'err0');

%%
% **********************************
%             TEST 0b
% **********************************

for i = 1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
    
    [idx_cls1, idx_cls2, idx_cls3] = findClusters(yTr, XTr);
    yTr_cls1 = yTr(idx_cls1,:);
    yTr_cls2 = yTr(idx_cls2,:);
    yTr_cls3 = yTr(idx_cls3,:);
    
    [idx_cls1, idx_cls2, idx_cls3] = findClusters(yTe, XTe);
    yTe_cls1 = yTe(idx_cls1,:);
    yTe_cls2 = yTe(idx_cls2,:);
    yTe_cls3 = yTe(idx_cls3,:);
    
    % Nothing to learn
    beta_cls1 = mean(yTr_cls1);
    beta_cls2 = mean(yTr_cls2);
    beta_cls3 = mean(yTr_cls3);
    
    err0b_cls1(i) = RMSE(yTe_cls1, beta_cls1);
    err0b_cls2(i) = RMSE(yTe_cls2, beta_cls2);
    err0b_cls3(i) = RMSE(yTe_cls3, beta_cls3);
end

saveFile(err0b_cls1, 'err0b_cls1');
saveFile(err0b_cls2, 'err0b_cls2');
saveFile(err0b_cls3, 'err0b_cls3');

%%
% **********************************
%             TEST 1
% **********************************

K = 10;

for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
    
    % Learning
    lambda = findLambda(K, yTr, XTr, 0, 1);
    
    tXTr = [ones(length(yTr),1) XTr];
    beta = ridgeRegression(yTr, tXTr, lambda);
    
    tXTe = [ones(length(yTe),1) XTe];
    err1(i) = RMSE(yTe, tXTe*beta);
end

saveFile(err1, 'err1');

%%
% **********************************
%             TEST 2
% **********************************

K = 10;

for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, normalizedData(X));
    
    % Learning
    lambda = findLambda(K, yTr, XTr, 0, 1);
    
    tXTr = [ones(length(yTr),1) XTr];
    beta = ridgeRegression(yTr, tXTr, lambda);
    
    tXTe = [ones(length(yTe),1) XTe];
    err2(i) = RMSE(yTe, tXTe*beta);
end

saveFile(err2, 'err2');

%%
s = [1 numberOfExperiments];

err0 = openFile('err0', s);
err0b = openFile('err0b_cls1', s) + openFile('err0b_cls2',s) + openFile('err0b_cls3',s);
err1 = openFile('err1', s);
err2 = openFile('err2', s);

figure;
boxplot([err0' err0b' err1' err2']);
legend(findobj(gca,'Tag','Box'),'0 Constant model','0b Constant model per cluster', '1 Least squares', '2 + Normalization');
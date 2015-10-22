% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;
close all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;
N = length(y);

[y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3] = preprocess(y, X);
d1 = 3;
d2 = 5;
d3 = 2;

l1 = 0.035;
l2 = 0.1298;
l3 = 0.7480;

k1 = 5;
k2 = 5;
k3 = 4;

K = 0:5:95;
K = K(2:end);
K = K./100;
for i = 1:1:length(K)
    k = K(i);
    
    [err_tr_cls1(:,i), err_te_cls1(:,i)] = modelLearningCurve(k, k1, d1, l1, y_cls1, X_cls1);
    [err_tr_cls2(:,i), err_te_cls2(:,i)] = modelLearningCurve(k, k2, d2, l2, y_cls2, X_cls2);
    [err_tr_cls3(:,i), err_te_cls3(:,i)] = modelLearningCurve(k, k3, d3, l3, y_cls3, X_cls3);
end

figure
boxplot(err_tr_cls1, 'labels', K);
hold on;
boxplot(err_te_cls1, 'labels', K);
title('CLS1');
xlabel('Training size');
ylabel('RMSE');
hold off;

figure
boxplot(err_tr_cls2, 'labels', K);
hold on;
boxplot(err_te_cls2, 'labels', K);
title('CLS2');
xlabel('Training size');
ylabel('RMSE');
hold off;

figure
boxplot(err_tr_cls3, 'labels', K);
hold on;
boxplot(err_te_cls3, 'labels', K);
title('CLS3');
xlabel('Training size');
ylabel('RMSE');
hold off;
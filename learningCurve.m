% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;
close all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;
N = length(y);

numberOfExperiments = 2;
proportionOfTraining = 0.8;

d1 = 3;
d2 = 5;
d3 = 2;

k1 = 10;
k2 = 6;
k3 = 4;

K = 10:10:90;
K = K(2:end);
K = K./100;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
    fprintf('%d\n', j); 

    for i = 1:1:length(K)
        k = K(i);
        [err_tr_cls1(j,i), err_te_cls1(j,i), err_tr_cls2(j,i), err_te_cls2(j,i), err_tr_cls3(j,i), err_te_cls3(j,i)] = modelLearningCurve(k, k1, k2, k3, yTr, XTr, d1, d2, d3, yTe, XTe);
    end
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
% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;

yN = normalizedData(y);
XN = normalizedData(X);
[y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3] = preprocess(y, X);

res = [0 0 0];
for i = 1:1:length(XN)
    idx_cls = findCorrespondingCluster(XN(i,:));
    res(idx_cls) = res(idx_cls)+1;
end
% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;

xNorm=X-mean(X(:));
xNorm=xNorm/std(xNorm(:));

N = length(y);
tX = [ones(N,1) xNorm];
  
alpha = 0.5;
beta = leastSquaresGD(y, tX, alpha);
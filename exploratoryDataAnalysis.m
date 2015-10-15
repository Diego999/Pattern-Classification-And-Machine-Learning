% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;
N = length(y);

figure
boxplot(X,'notch','on');

% Find outliers, we can say that outliers are bins where % is less than 1%.
figure
edges = 0:100:20000;
histogram(y,edges);
[nb_per_bins, bin_limits] = histcounts(y, edges);

% the value x_i2 corresponds to the nb in the bins [x_i1;x_(i+1)2]
%nb_per_bins = [nb_per_bins 0]';
%nb_per_bins = [bin_limits' nb_per_bins (nb_per_bins./N).*100.0];
% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;
close all;
load('Sydney_classification.mat');

y = y_train;
X = X_train;
N = length(y);

%% Plot of output
f1 = figure;
boxplot(X,'notch','on');
xlabel('Features');
ylabel('Output');
% saveas(f1, 'plots/general/boxplot_unnormalized.jpg');

%% Plot of normalized output
f2 = figure;
boxplot(normalizedData(X),'notch','on');
xlabel('Features');
ylabel('Output');
% saveas(f2, 'plots/general/boxplot_normalized.jpg');

%% Histogram of output
f3 = figure;
histogram(y);
xlabel('Y');
ylabel('Count');
xlim([-1 1]);
% saveas(f3, 'plots/general/histogram.jpg');

%% Plot feature VS Output
for feature=1:size(X,2)
    f = figure;
    plot(X(:,feature),y,'ob');
    ylabel('y');
    xlabel(sprintf('Values of feature %d', feature));
    % saveas(f, sprintf('plots/features/featuresVSOutput/feature%d.jpg', feature));
end

%% Plot histogram of each feature
for feature=1:size(X,2)
    f = figure;
    histogram(X(:,feature));
    xlabel(sprintf('Feature %d', feature));
    ylabel('Count');
    % saveas(f, sprintf('plots/features/featuresHistogram/feature%d.jpg', feature));
end

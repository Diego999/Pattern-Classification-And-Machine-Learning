% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;
close all;

load train/train.mat;

y = train.y;
X_cnn = train.X_cnn;
X_hog = train.X_hog;

% y(find(y == 1)) = 1;
% y(find(y == 2)) = 1;
% y(find(y == 3)) = 1;
% y(find(y == 4)) = 2;

%% Classes

lc1 = length(find(y == 1)); % 16.07%
lc2 = length(find(y == 2)); % 19.37%
lc3 = length(find(y == 3)); % 24.87%
lc4 = length(find(y == 4)); % 39.7%

%% Classes

lc1 = length(find(y == 1)); % 60.3%
lc2 = length(find(y == 2)); % 39.7%

%% HOG
% Explanation what is it
min_hog = min(unique(X_hog)); % 0
max_hog = max(unique(X_hog)); % 0.2
rank_hog = rank(X_hog); % 3467
corr_hog = corr(double(X_hog), double(y)); %[-0.243;0.3]

%% CNN
% Sparse !
% Explanation what is it

min_cnn = min(unique(X_cnn)); % 0
max_cnn = max(unique(X_cnn)); % 64.87
rank_cnn = rang(X_cnn); % 5997
corr_cnn = corr(double(X_cnn), double(y)); %[-0.2393;0.1795]

%% Histogram

f1 = figure;
histogram(y, 'Normalization','probability');
xlabel('Y');
ylabel('Count');
xlim([1 4]); 
set(gca,'fontsize', 18);
legend('1', '2');
hold off;
% saveas(f2, 'plots/cluster/feature16.jpg');
%print('./report/figures/distortionCNN.jpg','-djpeg','-noui');

%% Histogram feature HOG (distribution)

idx = find(y == 1);
X = X_hog;
for feature=1:10%size(X,2)
    f = figure;
    histogram(X(idx,feature));
    xlabel(sprintf('Feature %d', feature));
    ylabel('Count');
end

idx = find(y == 2);
X = X_hog;
for feature=1:10%size(X,2)
    f = figure;
    histogram(X(idx,feature));
    xlabel(sprintf('Feature %d', feature));
    ylabel('Count');
end

%% Histogram feature CNN (distribution)

idx = find(y == 1);
X = X_cnn;
for feature=1:10%size(X,2)
    f = figure;
    histogram(X(idx,feature));
    xlabel(sprintf('Feature %d', feature));
    ylabel('Count');
end

idx = find(y == 2);
X = X_cnn;
for feature=1:10%size(X,2)
    f = figure;
    histogram(X(idx,feature));
    xlabel(sprintf('Feature %d', feature));
    ylabel('Count');
end
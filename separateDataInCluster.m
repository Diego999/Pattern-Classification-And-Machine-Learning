% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;
N = length(y);
XN = normalizedData(X);
yN = normalizedData(y);

% Find first separation of cluster using feature 2
tX = [ones(length(y), 1) XN(:,2)];
beta = [0 0]';
f = tX*beta;

idx_cluster1 = find(yN < 0);
X_cluster1 = XN(idx_cluster1,:);
y_cluster1 = yN(idx_cluster1);

f1 = figure;
plot(X_cluster1(:,2), y_cluster1, 'ob');
ylim([min(yN) max(yN)]);
xlim([min(XN(:,2)) max(XN(:,2))]);
xlabel('Feature 2');
ylabel('Y');
hold on
idx_others_clusters = setdiff(1:length(y), idx_cluster1)';
X_others_clusters = XN(idx_others_clusters,:);
y_others_clusters = yN(idx_others_clusters);
plot(X_others_clusters(:, 2), y_others_clusters, 'or');
hold off;
% saveas(f1, 'plots/cluster/feature2.jpg');

% Find second separation of cluster using feature 16
tX = [ones(length(X_others_clusters), 1) X_others_clusters(:,16)];

beta = [0 0]';
f = tX*beta;

idx_cluster2 = find(yN < 1.65);
X_cluster2 = XN(idx_cluster2,:);
y_cluster2 = yN(idx_cluster2);

idx_cluster3 = setdiff(idx_others_clusters, idx_cluster2);
X_cluster3 = XN(idx_cluster3,:);
y_cluster3 = yN(idx_cluster3);

f2 = figure;
plot(X_cluster2(:,16), y_cluster2, 'or');
ylim([min(yN) max(yN)]);
xlim([min(XN(:,16)) max(XN(:,16))]);
xlabel('Feature 16');
ylabel('Y');
hold on
plot(X_cluster3(:, 16), y_cluster3, 'ob');
hold off;
% saveas(f2, 'plots/cluster/feature16.jpg');

idx_cluster2 = setdiff(idx_others_clusters, idx_cluster3);

% Some checks
assert(length(idx_cluster1) + length(idx_cluster2) + length(idx_cluster3) == length(X));

% Print histogram
f3 = figure;
edges = 0:50:20000;
h1 = histogram(y(idx_cluster1), edges);
h1.FaceColor = [0.0 0.0 1.0];
hold on;
h2 = histogram(y(idx_cluster2), edges);
h2.FaceColor = [1.0 0.0 0.0];
hold on;
h3 = histogram(y(idx_cluster3), edges);
h3.FaceColor = [0.0 1.0 0.0];
hold off;
xlabel('Y');
ylabel('Count');
% saveas(f3, 'plots/cluster/histogram.jpg');

% Print feature vs output with color of clusters
% for feature=1:size(X,2)
%     f = figure;
%     plot(X(idx_cluster1,feature),y(idx_cluster1),'ob');
%     hold on;
%     plot(X(idx_cluster2,feature),y(idx_cluster2),'or');
%     hold on;
%     plot(X(idx_cluster3,feature),y(idx_cluster3),'og');
%     hold off;
%     ylabel('y');
%     xlabel(sprintf('Values of feature %d', feature));
%     saveas(f, sprintf('plots/cluster/featuresVSOutput/feature%d.jpg', feature));
% end
% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;
N = length(y);

% f1 = figure;
% boxplot(X,'notch','on');
% xlabel('Features');
% ylabel('Output');
% %saveas(f1, 'plots/general/boxplot_unnormalized.jpg');
% 
% f2 = figure;
% boxplot(normalizedData(X),'notch','on');
% xlabel('Features');
% ylabel('Output');
% %saveas(f2, 'plots/general/boxplot_normalized.jpg');

% f3 = figure;
% edges = 0:50:20000;
% histogram(y,edges);
% xlabel('Y');
% ylabel('Count');
% saveas(f3, 'plots/general/histogram.jpg');
% [nb_per_bins, bin_limits] = histcounts(y, edges);
% 
% % the value x_i2 corresponds to the nb in the bins [x_i1;x_(i+1)2]
% nb_per_bins = [nb_per_bins 0]';
% nb_per_bins = [bin_limits' nb_per_bins (nb_per_bins./N).*100.0];

% for feature=1:size(X,2)
%     f = figure;
%     plot(X(:,feature),y,'ob');
%     ylabel('y');
%     xlabel(sprintf('Values of feature %d', feature));
%     saveas(f, sprintf('plots/features/featuresVSOutput/feature%d.jpg', feature));
% end

% Plot feature2 and 16 against others
% x_features = [2 16];
% for x_f_idx=1:length(x_features)
%     x_f = x_features(x_f_idx);
%     for feature=1:size(X,2)
%         if(feature==x_f)
%             continue;
%         end
%         f = figure;
%         plot(X(:,x_f),X(:,feature),'ob');
%         ylabel(sprintf('Values of feature %d', feature));
%         xlabel(sprintf('Values of feature %d', x_f));
%         saveas(f, sprintf('plots/features/feature%dVSAll/feature%d.jpg', x_f, feature));
%     end
% end

% Plot histogram of each feature
% for feature=1:size(X,2)
%     f = figure;
%     histogram(X(:,feature));
%     xlabel(sprintf('Feature %d', feature));
%     ylabel('Count');
%     saveas(f, sprintf('plots/features/featuresHistogram/feature%d.jpg', feature));
% end

% Show correlation between feature X to Y
% f = figure;
% corr_features = corr(X,y);
% x_domain = 1:1:size(X,2);
% plot(x_domain, corr_features, 'ob');
% hold on
% plot(x_domain, x_domain*0, '-r');
% xlabel('Feature i');
% ylabel('Correlation between feature i and y');
% saveas(f, sprintf('plots/features/featuresCorrelationOutput.jpg'));
    
% % Find clusers and remove outliers
% XN = normalizedData(X);
% yN = normalizedData(y);
% 
% idx_cluster1 = find(XN(:,2) < 0.42);
% X_cluster1 = XN(idx_cluster1,:);
% y_cluster1 = yN(idx_cluster1);
% 
% idx_others_clusters = setdiff(1:length(y), idx_cluster1)';
% X_others_clusters = XN(idx_others_clusters,:);
% y_others_clusters = yN(idx_others_clusters);
% 
% % There are 12 outliers here !
% outliers_cluster1 = find(yN > 3.5 | (XN(:,2) < 0.42 & yN > 0.4) | (XN(:,2) > 0.42 & yN < 0));
% 
% % Draw clusters with outliers
% f1 = figure;
% 
% ylim([min(yN) max(yN)]);
% xlim([min(XN(:,2)) max(XN(:,2))]);
% xlabel('Feature 2');
% ylabel('Y');
% plot(XN(idx_cluster1,2), yN(idx_cluster1), 'ob');
% hold on
% plot(X_others_clusters(:, 2), y_others_clusters, 'or');
% hold on
% plot(XN(outliers_cluster1, 2), yN(outliers_cluster1), 'oy');
% hold off;
% % saveas(f1, 'plots/cluster/feature2.jpg');
% 
% % Find second separation of cluster using feature 16
% idx_cluster3 = setdiff(idx_others_clusters, find(XN(:,16) < 1.17));
% 
% % There is 11 outlier here
% outliers_cluster3 = find(yN > 2 & XN(:,16) < 1.17);
% 
% X_cluster3 = XN(idx_cluster3,:);
% y_cluster3 = yN(idx_cluster3);
% 
% idx_cluster2 = setdiff(idx_others_clusters, idx_cluster3);
% X_cluster2 = XN(idx_cluster2,:);
% y_cluster2 = yN(idx_cluster2);
% 
% % There are 5 outliers here !
% outliers_cluster2 = find(yN > 3.5 | (XN(:,16) > 1.17 & yN > 0.5 & yN < 1.5));
% 
% f2 = figure;
% ylim([min(yN) max(yN)]);
% xlim([min(XN(:,16)) max(XN(:,16))]);
% xlabel('Feature 16');
% ylabel('Y');
% hold on
% plot(X_cluster1(:,16), y_cluster1, 'ob');
% hold on;
% plot(X_cluster2(:,16), y_cluster2, 'or');
% hold on
% plot(X_cluster3(:,16), y_cluster3, 'og');
% hold on
% plot(XN(outliers_cluster3, 16), yN(outliers_cluster3), 'oy');
% hold on;
% plot(XN(outliers_cluster2, 16), yN(outliers_cluster2), 'oy');
% hold on
% plot(XN(outliers_cluster1, 16), yN(outliers_cluster1), 'oy');
% hold off;
% saveas(f2, 'plots/cluster/feature16.jpg');
% 
% % Some checks
% assert(length(idx_cluster1) + length(idx_cluster2) + length(idx_cluster3) == length(X));
% 
% idx_cls1 = setdiff(setdiff(setdiff(idx_cluster1, outliers_cluster1), outliers_cluster2), outliers_cluster3);
% idx_cls2 = setdiff(setdiff(setdiff(idx_cluster2, outliers_cluster1), outliers_cluster2), outliers_cluster3);
% idx_cls3 = setdiff(setdiff(setdiff(idx_cluster3, outliers_cluster1), outliers_cluster2), outliers_cluster3);
% 
% % Print histogram
% f3 = figure;
% edges = 0:50:20000;
% h1 = histogram(y(idx_cls1), edges);
% h1.FaceColor = [0.0 0.0 1.0];
% hold on;
% h2 = histogram(y(idx_cls2), edges);
% h2.FaceColor = [1.0 0.0 0.0];
% hold on;
% h3 = histogram(y(idx_cls3), edges);
% h3.FaceColor = [0.0 1.0 0.0];
% hold off;
% xlabel('Y');
% ylabel('Count');
% saveas(f3, 'plots/cluster/histogram_without_outliers.jpg');

% % Print feature vs output with color of clusters
% for feature=1:size(X,2)
%     f = figure;
%     plot(X(idx_cls1,feature),y(idx_cls1),'ob');
%     hold on;
%     plot(X(idx_cls2,feature),y(idx_cls2),'or');
%     hold on;
%     plot(X(idx_cls3,feature),y(idx_cls3),'og');
%     hold off;
%     ylabel('y');
%     xlabel(sprintf('Values of feature %d', feature));
%     saveas(f, sprintf('plots/cluster/featuresVSOutput/feature%d.jpg', feature));
% end
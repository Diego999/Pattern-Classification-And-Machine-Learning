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
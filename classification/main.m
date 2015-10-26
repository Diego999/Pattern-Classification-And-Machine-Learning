% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;
close all;

load('Sydney_classification.mat');

y = y_train;
X = X_train;
N = length(y);

% 1) Constant baseline
% 2) Logistic regression
% 3) + Output either 0 or 1
% 4) + without categorical variables
% 5) + Removing outliers

numberOfExperiments = 30;
proportionOfTraining = 0.8;

%%
% **********************************
%             TEST 1
% **********************************

for i = 1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    out = 1;
    
    err1(i) = RMSE(yTe, out);
    err1_01(i) = zeroOneLoss(yTe, repmat([out], length(yTr), 1));
    err1_log(i) = logLoss(yTe, repmat([out], length(yTr), 1)*0.9999999); % To avoid log of 0
end

saveFile(err1, 'results/err1');
saveFile(err1_01, 'results/err1_01');
saveFile(err1_log, 'results/err1_log');

%%
s = [1 numberOfExperiments];

err1 = openFile('results/err1', s);
err1_01 = openFile('results/err1_01', s);
err1_log = openFile('results/err1_log', s);
% 1) Constant baseline

% RMSE
figure;
boxplot([err1']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 Constant model');
set(h_legend,'FontSize',12);
set(gca, 'XGrid','on')
set(gca, 'YGrid','on')
%ylim([0 3000])
%set(gca,'YTick',0:100:3000)
xlabel('Model');
ylabel('RMSE');
%print('../report/figures/models','-djpeg','-noui')

%0-1Loss
figure;
boxplot([err1_01']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 Constant model');
set(h_legend,'FontSize',12);
set(gca, 'XGrid','on')
set(gca, 'YGrid','on')
%ylim([0 3000])
%set(gca,'YTick',0:100:3000)
xlabel('Model');
ylabel('0-1 Loss');
%print('../report/figures/models','-djpeg','-noui')

%LogLoss
figure;
boxplot([err1_log']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 Constant model');
set(h_legend,'FontSize',12);
set(gca, 'XGrid','on')
set(gca, 'YGrid','on')
%ylim([0 3000])
%set(gca,'YTick',0:100:3000)
xlabel('Model');
ylabel('Log Loss');
%print('../report/figures/models','-djpeg','-noui')
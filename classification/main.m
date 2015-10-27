% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;
close all;

load('Sydney_classification.mat');

y = y_train;
X = X_train;
N = length(y);

% 1) Constant baseline
% 2) Logistic regression (with normalization + output [0,1] + keaping only real features)
% 3) + categorical variables
% 4) + dummy encoding
% 5) + Removing outliers

numberOfExperiments = 30;
proportionOfTraining = 0.8;
K = 10;

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
% **********************************
%             TEST 2
% **********************************
% We MUST normalize and transfomr -1 to 0 in order to have GD working

alphas = [0.00075];
for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    [yTr, XTr] = preprocess(yTr, XTr);
    N = length(yTr);
    idxCV = splitGetCV(K, N);
    
    for a = 1:1:length(alphas)
        % K-fold
        for k=1:1:K
            [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
            
            tXTr = [ones(length(yyTr),1) XXTr];
            tXTe = [ones(length(yyTe),1) XXTe];
            
            alpha = alphas(a);
            beta = logisticRegression(yyTr, tXTr, alpha);
            
            err_tr_rr(k,a) = RMSE(yyTr, tXTr*beta);
            err_te_rr(k,a) = RMSE(yyTe, tXTe*beta);
        end

        mseTr = mean(err_tr_rr);
        mseTe = mean(err_te_rr);
    end
    
    [errStar, alphaStarId] = min(mseTe);
    alphaStar = alphas(alphaStarId);
    
    [yTe, XTe] = preprocess(yTe, XTe);
    
    tXTe = [ones(length(yTe),1) XTe];
    
    beta = logisticRegression(yTe, tXTe, alphaStar);
    y_hat = (sigmoid(tXTe*beta) >= 0.5).*1.0;
    
    err2(i) = RMSE(yTe, y_hat);
    err2_01(i) = zeroOneLoss(yTe, y_hat);
    err2_log(i) = logLoss(yTe, y_hat*0.9999999); % To avoid log of 0
end

saveFile(err2, 'results/err2');
saveFile(err2_01, 'results/err2_01');
saveFile(err2_log, 'results/err2_log');


%%
s = [1 numberOfExperiments];

err1 = openFile('results/err1', s);
err1_01 = openFile('results/err1_01', s);
err1_log = openFile('results/err1_log', s);

err2 = openFile('results/err2', s);
err2_01 = openFile('results/err2_01', s);
err2_log = openFile('results/err2_log', s);

% 1) Constant baseline
% 2) Logistic regression (with normalization + output [0,1])

% RMSE
figure;
boxplot([err1' err2']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 Constant model', ...
'2 Logistic regression with normalization and output[0,1]');
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
boxplot([err1_01' err2_01']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 Constant model', ...
'2 Logistic regression with normalization and output[0,1]');
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
boxplot([err1_log' err2_log']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 Constant model', ...
'2 Logistic regression with normalization and output[0,1]');
set(h_legend,'FontSize',12);
set(gca, 'XGrid','on')
set(gca, 'YGrid','on')
%ylim([0 3000])
%set(gca,'YTick',0:100:3000)
xlabel('Model');
ylabel('Log Loss');
%print('../report/figures/models','-djpeg','-noui')
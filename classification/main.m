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
% 5) + Polynomial basis 2
% 6) Feature transformation (abs) (without polynomial basis)
% 7) Feature transformation (abs) (with polynomial basis 2)
% 8) Feature transformation (with -) (without polynomial basis)
% 9) Feature transformation (with -) (with polynomial basis 2)

% 6) + Polynomial basis degree 3
% 5) + Removing outliers

numberOfExperiments = 30;
proportionOfTraining = 0.8;
K = 10;

%%
%**********************************
%            TEST 1
%**********************************

for i = 1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    out = 1;
    
    err1(i) = RMSE(yTe, out);
    err1_01(i) = zeroOneLoss(yTe, repmat([out], length(yTr), 1));
    err1_log(i) = logLoss(yTe, repmat([out], length(yTr), 1));
end

saveFile(err1, 'results/err1');
saveFile(err1_01, 'results/err1_01');
saveFile(err1_log, 'results/err1_log');

%%
% **********************************
%             TEST 2
% **********************************
% We MUST normalize and transform -1 to 0 in order to have GD working
% Remove categorical features from preprocess

alphas = [0.00075];
for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    [yTr, XTr] = preprocess(yTr, XTr);
%     N = length(yTr);
%     idxCV = splitGetCV(K, N);
%     
%     for a = 1:1:length(alphas)
%         % K-fold
%         for k=1:1:K
%             [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
%             
%             tXTr = [ones(length(yyTr),1) XXTr];
%             tXTe = [ones(length(yyTe),1) XXTe];
%             
%             alpha = alphas(a);
%             beta = logisticRegression(yyTr, tXTr, alpha);
%             
%             err_tr_rr(k,a) = RMSE(yyTr, tXTr*beta);
%             err_te_rr(k,a) = RMSE(yyTe, tXTe*beta);
%         end
% 
%         mseTr = mean(err_tr_rr);
%         mseTe = mean(err_te_rr);
%     end
%     
%     [errStar, alphaStarId] = min(mseTe);
%     alphaStar = alphas(alphaStarId);
    alphaStar = alphas(1);
    [yTe, XTe] = preprocess(yTe, XTe);
    
    tXTe = [ones(length(yTe),1) XTe];
    tXTr = [ones(length(yTr),1) XTr];
    
    beta = logisticRegression(yTr, tXTr, alphaStar);
    y_hat = (sigmoid(tXTe*beta) > 0.5).*1.0;
    
    err2(i) = RMSE(yTe, y_hat);
    err2_01(i) = zeroOneLoss(yTe, y_hat);
    err2_log(i) = logLoss(yTe, y_hat);
end

saveFile(err2, 'results/err2');
saveFile(err2_01, 'results/err2_01');
saveFile(err2_log, 'results/err2_log');

%%
% **********************************
%             TEST 3
% **********************************
% We MUST normalize and transform -1 to 0 in order to have GD working
% Add categorical features from preprocess

alphas = [0.0005];
for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    [yTr, XTr] = preprocess(yTr, XTr);
%     N = length(yTr);
%     idxCV = splitGetCV(K, N);
%     
%     for a = 1:1:length(alphas)
%         % K-fold
%         for k=1:1:K
%             [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
%             
%             tXTr = [ones(length(yyTr),1) XXTr];
%             tXTe = [ones(length(yyTe),1) XXTe];
%             
%             alpha = alphas(a);
%             beta = logisticRegression(yyTr, tXTr, alpha);
%             
%             err_tr_rr(k,a) = RMSE(yyTr, tXTr*beta);
%             err_te_rr(k,a) = RMSE(yyTe, tXTe*beta);
%         end
% 
%         mseTr = mean(err_tr_rr);
%         mseTe = mean(err_te_rr);
%     end
%     
%     [errStar, alphaStarId] = min(mseTe);
%     alphaStar = alphas(alphaStarId);
    alphaStar = alphas(1);
    
    [yTe, XTe] = preprocess(yTe, XTe);
    
    tXTe = [ones(length(yTe),1) XTe];
    tXTr = [ones(length(yTr),1) XTr];
    
    beta = logisticRegression(yTr, tXTr, alphaStar);
    y_hat = (sigmoid(tXTe*beta) >= 0.5).*1.0;
    
    err3(i) = RMSE(yTe, y_hat);
    err3_01(i) = zeroOneLoss(yTe, y_hat);
    err3_log(i) = logLoss(yTe, y_hat);
end

saveFile(err3, 'results/err3');
saveFile(err3_01, 'results/err3_01');
saveFile(err3_log, 'results/err3_log');

%%
% **********************************
%             TEST 4
% **********************************
% We MUST normalize and transform -1 to 0 in order to have GD working
% Add categorical features from preprocess
% Add dummy encoding

alphas = [0.0005];
for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    [yTr, XTr] = preprocess(yTr, XTr);
    N = length(yTr);
    idxCV = splitGetCV(K, N);
    
%      for a = 1:1:length(alphas)
%          % K-fold
%          for k=1:1:K
%              [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
%              
%              tXTr = [ones(length(yyTr),1) XXTr];
%              tXTe = [ones(length(yyTe),1) XXTe];
%              
%              alpha = alphas(a);
%              beta = logisticRegression(yyTr, tXTr, alpha);
%              
%              err_tr_rr(k,a) = RMSE(yyTr, tXTr*beta);
%              err_te_rr(k,a) = RMSE(yyTe, tXTe*beta);
%          end
%  
%          mseTr = mean(err_tr_rr);
%          mseTe = mean(err_te_rr);
%      end
    
    beta = logisticRegression(yTr, [ones(length(yTr), 1) XTr], alphas(1));
    %[errStar, alphaStarId] = min(mseTe);
    %alphaStar = alphas(alphaStarId);
    
    [yTe, XTe] = preprocess(yTe, XTe);
    
    tXTe = [ones(length(yTe),1) XTe];
    
    %beta = logisticRegression(yTe, tXTe, alphaStar);
    y_hat = (sigmoid(tXTe*beta) >= 0.5).*1.0;
    
    err4(i) = RMSE(yTe, y_hat);
    err4_01(i) = zeroOneLoss(yTe, y_hat);
    err4_log(i) = logLoss(yTe, y_hat);
end

%saveFile(err4, 'results/err4');
%saveFile(err4_01, 'results/err4_01');
%saveFile(err4_log, 'results/err4_log');

%%
% **********************************
%             TEST 5
% **********************************
% We MUST normalize and transform -1 to 0 in order to have GD working
% Add categorical features from preprocess
% Add dummy encoding
% Add polynomial basis

alphas = [0.001];
degree = 2;
for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    [yTr, XTr] = preprocess(yTr, XTr, degree);
    N = length(yTr);
    idxCV = splitGetCV(K, N);
    
%      for a = 1:1:length(alphas)
%          % K-fold
%          for k=1:1:K
%              [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
%              
%              tXTr = [ones(length(yyTr),1) XXTr];
%              tXTe = [ones(length(yyTe),1) XXTe];
%              
%              alpha = alphas(a);
%              beta = logisticRegression(yyTr, tXTr, alpha);
%              
%              err_tr_rr(k,a) = RMSE(yyTr, tXTr*beta);
%              err_te_rr(k,a) = RMSE(yyTe, tXTe*beta);
%          end
%  
%          mseTr = mean(err_tr_rr);
%          mseTe = mean(err_te_rr);
%      end
    
    beta = logisticRegression(yTr, [ones(length(yTr), 1) XTr], alphas(1));
    %[errStar, alphaStarId] = min(mseTe);
    %alphaStar = alphas(alphaStarId);
    
    [yTe, XTe] = preprocess(yTe, XTe, degree);
    
    tXTe = [ones(length(yTe),1) XTe];
    
    %beta = logisticRegression(yTe, tXTe, alphaStar);
    y_hat = (sigmoid(tXTe*beta) >= 0.5).*1.0;
    
    err5(i) = RMSE(yTe, y_hat);
    err5_01(i) = zeroOneLoss(yTe, y_hat);
    err5_log(i) = logLoss(yTe, y_hat);
end

saveFile(err5, 'results/err5');
saveFile(err5_01, 'results/err5_01');
saveFile(err5_log, 'results/err5_log');

%%
% **********************************
%             TEST 6
% **********************************
% We MUST normalize and transform -1 to 0 in order to have GD working
% Add categorical features from preprocess
% Add dummy encoding
% Add Transformation

alphas = [0.001];
for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    [yTr, XTr] = preprocess(yTr, XTr);
    N = length(yTr);
    idxCV = splitGetCV(K, N);
    
%      for a = 1:1:length(alphas)
%          % K-fold
%          for k=1:1:K
%              [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
%              
%              tXTr = [ones(length(yyTr),1) XXTr];
%              tXTe = [ones(length(yyTe),1) XXTe];
%              
%              alpha = alphas(a);
%              beta = logisticRegression(yyTr, tXTr, alpha);
%              
%              err_tr_rr(k,a) = RMSE(yyTr, tXTr*beta);
%              err_te_rr(k,a) = RMSE(yyTe, tXTe*beta);
%          end
%  
%          mseTr = mean(err_tr_rr);
%          mseTe = mean(err_te_rr);
%      end
    
    beta = logisticRegression(yTr, [ones(length(yTr), 1) XTr], alphas(1));
    %[errStar, alphaStarId] = min(mseTe);
    %alphaStar = alphas(alphaStarId);
    
    [yTe, XTe] = preprocess(yTe, XTe);
    
    tXTe = [ones(length(yTe),1) XTe];
    
    %beta = logisticRegression(yTe, tXTe, alphaStar);
    y_hat = (sigmoid(tXTe*beta) >= 0.5).*1.0;
    
    err6(i) = RMSE(yTe, y_hat);
    err6_01(i) = zeroOneLoss(yTe, y_hat);
    err6_log(i) = logLoss(yTe, y_hat);
end

saveFile(err6, 'results/err6');
saveFile(err6_01, 'results/err6_01');
saveFile(err6_log, 'results/err6_log');

%%
% **********************************
%             TEST 7
% **********************************
% We MUST normalize and transform -1 to 0 in order to have GD working
% Add categorical features from preprocess
% Add dummy encoding
% Add polynomial basis
% Add Transformation

alphas = [0.001];
degree = 2;
for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    [yTr, XTr] = preprocess(yTr, XTr, degree);
    N = length(yTr);
    idxCV = splitGetCV(K, N);
    
%      for a = 1:1:length(alphas)
%          % K-fold
%          for k=1:1:K
%              [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
%              
%              tXTr = [ones(length(yyTr),1) XXTr];
%              tXTe = [ones(length(yyTe),1) XXTe];
%              
%              alpha = alphas(a);
%              beta = logisticRegression(yyTr, tXTr, alpha);
%              
%              err_tr_rr(k,a) = RMSE(yyTr, tXTr*beta);
%              err_te_rr(k,a) = RMSE(yyTe, tXTe*beta);
%          end
%  
%          mseTr = mean(err_tr_rr);
%          mseTe = mean(err_te_rr);
%      end
    
    beta = logisticRegression(yTr, [ones(length(yTr), 1) XTr], alphas(1));
    %[errStar, alphaStarId] = min(mseTe);
    %alphaStar = alphas(alphaStarId);
    
    [yTe, XTe] = preprocess(yTe, XTe, degree);
    
    tXTe = [ones(length(yTe),1) XTe];
    
    %beta = logisticRegression(yTe, tXTe, alphaStar);
    y_hat = (sigmoid(tXTe*beta) >= 0.5).*1.0;
    
    err7(i) = RMSE(yTe, y_hat);
    err7_01(i) = zeroOneLoss(yTe, y_hat);
    err7_log(i) = logLoss(yTe, y_hat);
end

saveFile(err7, 'results/err7');
saveFile(err7_01, 'results/err7_01');
saveFile(err7_log, 'results/err7_log');

%%
% **********************************
%             TEST 8
% **********************************
% We MUST normalize and transform -1 to 0 in order to have GD working
% Add categorical features from preprocess
% Add dummy encoding
% Add Transformation (with -)

alphas = [0.001];
for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    [yTr, XTr] = preprocess(yTr, XTr);
    N = length(yTr);
    idxCV = splitGetCV(K, N);
    
%      for a = 1:1:length(alphas)
%          % K-fold
%          for k=1:1:K
%              [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
%              
%              tXTr = [ones(length(yyTr),1) XXTr];
%              tXTe = [ones(length(yyTe),1) XXTe];
%              
%              alpha = alphas(a);
%              beta = logisticRegression(yyTr, tXTr, alpha);
%              
%              err_tr_rr(k,a) = RMSE(yyTr, tXTr*beta);
%              err_te_rr(k,a) = RMSE(yyTe, tXTe*beta);
%          end
%  
%          mseTr = mean(err_tr_rr);
%          mseTe = mean(err_te_rr);
%      end
    
    beta = logisticRegression(yTr, [ones(length(yTr), 1) XTr], alphas(1));
    %[errStar, alphaStarId] = min(mseTe);
    %alphaStar = alphas(alphaStarId);
    
    [yTe, XTe] = preprocess(yTe, XTe);
    
    tXTe = [ones(length(yTe),1) XTe];
    
    %beta = logisticRegression(yTe, tXTe, alphaStar);
    y_hat = (sigmoid(tXTe*beta) >= 0.5).*1.0;
    
    err8(i) = RMSE(yTe, y_hat);
    err8_01(i) = zeroOneLoss(yTe, y_hat);
    err8_log(i) = logLoss(yTe, y_hat);
end

saveFile(err8, 'results/err8');
saveFile(err8_01, 'results/err8_01');
saveFile(err8_log, 'results/err8_log');

%%
% **********************************
%             TEST 9
% **********************************
% We MUST normalize and transform -1 to 0 in order to have GD working
% Add categorical features from preprocess
% Add dummy encoding
% Add polynomial basis
% Add Transformation (with -)

alphas = [0.001075];
degree = 3;
for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    [yTr, XTr] = preprocess(yTr, XTr, degree);
    N = length(yTr);
    idxCV = splitGetCV(K, N);
    
%      for a = 1:1:length(alphas)
%          % K-fold
%          for k=1:1:K
%              [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
%              
%              tXTr = [ones(length(yyTr),1) XXTr];
%              tXTe = [ones(length(yyTe),1) XXTe];
%              
%              alpha = alphas(a);
%              beta = logisticRegression(yyTr, tXTr, alpha);
%              
%              err_tr_rr(k,a) = RMSE(yyTr, tXTr*beta);
%              err_te_rr(k,a) = RMSE(yyTe, tXTe*beta);
%          end
%  
%          mseTr = mean(err_tr_rr);
%          mseTe = mean(err_te_rr);
%      end
    
    beta = logisticRegression(yTr, [ones(length(yTr), 1) XTr], alphas(1));
    %[errStar, alphaStarId] = min(mseTe);
    %alphaStar = alphas(alphaStarId);
    
    [yTe, XTe] = preprocess(yTe, XTe, degree);
    
    tXTe = [ones(length(yTe),1) XTe];
    
    %beta = logisticRegression(yTe, tXTe, alphaStar);
    y_hat = (sigmoid(tXTe*beta) >= 0.5).*1.0;
    
    err9(i) = RMSE(yTe, y_hat);
    err9_01(i) = zeroOneLoss(yTe, y_hat);
    err9_log(i) = logLoss(yTe, y_hat);
end

saveFile(err9, 'results/err9');
saveFile(err9_01, 'results/err9_01');
saveFile(err9_log, 'results/err9_log');

%%
% **********************************
%             TEST 10
% **********************************
% We MUST normalize and transform -1 to 0 in order to have GD working
% Add categorical features from preprocess
% Add dummy encoding
% Add polynomial basis

%a = 0.001, d = 2 -> 90.53 (Without discrete variable and using abs)
alphas = [0.001];
degree = 2;
for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    [yTr, XTr] = preprocess(yTr, XTr, degree);
    N = length(yTr);
    idxCV = splitGetCV(K, N);
    
%      for a = 1:1:length(alphas)
%          % K-fold
%          for k=1:1:K
%              [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
%              
%              tXTr = [ones(length(yyTr),1) XXTr];
%              tXTe = [ones(length(yyTe),1) XXTe];
%              
%              alpha = alphas(a);
%              beta = logisticRegression(yyTr, tXTr, alpha);
%              
%              err_tr_rr(k,a) = RMSE(yyTr, tXTr*beta);
%              err_te_rr(k,a) = RMSE(yyTe, tXTe*beta);
%          end
%  
%          mseTr = mean(err_tr_rr);
%          mseTe = mean(err_te_rr);
%      end
    
    beta = logisticRegression(yTr, [ones(length(yTr), 1) XTr], alphas(1));
    %[errStar, alphaStarId] = min(mseTe);
    %alphaStar = alphas(alphaStarId);
    
    [yTe, XTe] = preprocess(yTe, XTe, degree);
    
    tXTe = [ones(length(yTe),1) XTe];
    
    %beta = logisticRegression(yTe, tXTe, alphaStar);
    y_hat = (sigmoid(tXTe*beta) >= 0.5).*1.0;
    
    err10(i) = RMSE(yTe, y_hat);
    err10_01(i) = zeroOneLoss(yTe, y_hat);
    err10_log(i) = logLoss(yTe, y_hat);
end

saveFile(err10, 'results/err10');
saveFile(err10_01, 'results/err10_01');
saveFile(err10_log, 'results/err10_log');

%%
s = [1 numberOfExperiments];

err1 = openFile('results/err1', s);
err1_01 = openFile('results/err1_01', s);
err1_log = openFile('results/err1_log', s);

err2 = openFile('results/err2', s);
err2_01 = openFile('results/err2_01', s);
err2_log = openFile('results/err2_log', s);

err3 = openFile('results/err3', s);
err3_01 = openFile('results/err3_01', s);
err3_log = openFile('results/err3_log', s);

err4 = openFile('results/err4', s);
err4_01 = openFile('results/err4_01', s);
err4_log = openFile('results/err4_log', s);

err5 = openFile('results/err5', s);
err5_01 = openFile('results/err5_01', s);
err5_log = openFile('results/err5_log', s);

err6 = openFile('results/err6', s);
err6_01 = openFile('results/err6_01', s);
err6_log = openFile('results/err6_log', s);

err7 = openFile('results/err7', s);
err7_01 = openFile('results/err7_01', s);
err7_log = openFile('results/err7_log', s);

err8 = openFile('results/err8', s);
err8_01 = openFile('results/err8_01', s);
err8_log = openFile('results/err8_log', s);

err9 = openFile('results/err9', s);
err9_01 = openFile('results/err9_01', s);
err9_log = openFile('results/err9_log', s);

err10 = openFile('results/err10', s);
err10_01 = openFile('results/err10_01', s);
err10_log = openFile('results/err10_log', s);

% 1) Constant baseline
% 2) Logistic regression (with normalization + output [0,1])
% 3) + categorical variables
% 4) + dummy encoding
% 5) + polynomial basis 2nd degree
% 6) Feature transformation (abs) (without polynomial basis)
% 7) Feature transformation (abs) (with polynomial basis)
% 8) Feature transformation (with -) (without polynomial basis)
% 9) Feature transformation (with -) (with polynomial basis)
% 10) Feature transformation (abs) (with polynomial basis) (without
% discrete)

% RMSE
figure;
boxplot([err1' err2' err3' err4' err5' err6' err7' err8' err9' err10']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 Constant model', ...
'2 Logistic regression with normalization and output[0,1]', ...
'3 + categorical features', ...
'4 + dummy encoding', ...
'5) + polynomial basis 2nd degree', ...
'6) Features transformation (abs), without polynomial basis', ...
'7) Feature transformation (abs) (with polynomial basis)', ...
'8) Features transformation (with -), without polynomial basis', ...
'9) Feature transformation (with -) (with polynomial basis)', ...
'10) Feature transformation (abs) (with polynomial basis) (without discrete)');
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
boxplot([err1_01' err2_01' err3_01' err4_01' err5_01' err6_01' err7_01' err8_01' err9_01' err10_01']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 Constant model', ...
'2 Logistic regression with normalization and output[0,1]', ...
'3 + categorical features', ...
'4 + dummy encoding', ...
'5) + polynomial basis 2nd degree', ...
'6) Features transformation (abs), without polynomial basis', ...
'7) Feature transformation (abs) (with polynomial basis)', ...
'8) Features transformation (with -), without polynomial basis', ...
'9) Feature transformation (with -) (with polynomial basis)', ...
'10) Feature transformation (abs) (with polynomial basis) (without discrete)');
set(h_legend,'FontSize',12);
set(gca, 'XGrid','on')
set(gca, 'YGrid','on')
%ylim([0 3000])
%set(gca,'YTick',0:100:3000)
xlabel('Model');
ylabel('0-1 Loss');
print('../report/figures/models_classification','-djpeg','-noui')

%LogLoss
figure;
boxplot([err1_log' err2_log' err3_log' err4_log' err5_log' err6_log' err7_log' err8_log' err9_log' err10_log']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 Constant model', ...
'2 Logistic regression with normalization and output[0,1]', ...
'3 + categorical features', ...
'4 + dummy encoding', ...
'5) + polynomial basis 2nd degree', ...
'6) Features transformation (abs), without polynomial basis', ...
'7) Feature transformation (abs) (with polynomial basis)', ...
'8) Features transformation (with -), without polynomial basis', ...
'9) Feature transformation (with -) (with polynomial basis)', ...
'10) Feature transformation (abs) (with polynomial basis) (without discrete)');
set(h_legend,'FontSize',12);
set(gca, 'XGrid','on')
set(gca, 'YGrid','on')
%ylim([0 3000])
%set(gca,'YTick',0:100:3000)
xlabel('Model');
ylabel('Log Loss');
%print('../report/figures/models','-djpeg','-noui')
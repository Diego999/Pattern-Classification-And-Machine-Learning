% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;
N = length(y);
[y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3] = preprocess(y, X);

% Methodology
% LS, won't work well due to ill-conditioning
% LSGD
% Ridge

% 1) Without any preprocess (no dummy encoding, no cluster separation, no normalization)
% 2) Same as 1) but with normalization
% 3) Separation with the cluster and normalization per cluster
% 4) Same as 3 but with dummy encoding
% 5) Polynomial basis
% 6) Remove some uncorrelated data

% **********************************
%             TEST 1
% **********************************
% K = 7;
% idxCV = splitGetCV(K, N);
% 
% for k=1:1:K
%     [XTr, yTr, XTe, yTe] = splitGetTrTe(y, X, idxCV, k);
%     
%     tXTr = [ones(length(yTr),1) XTr];
%     tXTe = [ones(length(yTe),1) XTe];
%     
%     % mean err1_tr_ls = 1219
%     % mean err1_te_ls = 1250
%     % Matrix ill conditionned, so very bad error
% %     beta1_ls = leastSquares(yTr, tXTr);
% %     err1_tr_ls(k) = RMSE(yTr, tXTr*beta1_ls);
% %     err1_te_ls(k) = RMSE(yTe, tXTe*beta1_ls);
%     
%     % mean err1_tr_lsgd : 1219
%     % mean err1_te_lsgd : 1250
% %     alpha = 0.002875; %0.0029 without CV
% %     beta1_lsgd = leastSquaresGD(yTr, tXTr, alpha);
% %     err1_tr_lsgd(k) = RMSE(yTr, tXTr*beta1_lsgd);
% %     err1_te_lsgd(k) = RMSE(yTe, tXTe*beta1_lsgd);
% 
% %     lambdas = logspace(-5,5,10000);
% %     for i = 1:1:length(lambdas)
% %         lambda = lambdas(i);
% %         beta1_rr = ridgeRegression(yTr, tXTr, lambda);
% %         err1_tr_rr(k,i) = RMSE(yTr, tXTr*beta1_rr);
% %         err1_te_rr(k,i) = RMSE(yTe, tXTe*beta1_rr);
% %     end
% 
%     % mean err1_tr_rr : 1219
%     % mean err1_te_rr : 1250
%     lambda = 0.001;
%     beta1_rr = ridgeRegression(yTr, tXTr, lambda);
%     err1_tr_rr(k) = RMSE(yTr, tXTr*beta1_rr);
%     err1_te_rr(k) = RMSE(yTe, tXTe*beta1_rr);
% end
% 
% % figure
% % mseTr = mean(err1_tr_rr);
% % mseTe = mean(err1_te_rr);
% % 
% % [errStar, lambdaIdStar] = min(mseTe);
% % lambdaStar = lambdas(lambdaIdStar);
% %     
% % semilogx(lambdas, mseTr, 'b-o', lambdas, mseTe, 'r-x');
% % hold on;
% % legend('Train error', 'Test error', 'Location', 'southeast');
% % semilogx(lambdaStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
% % xlabel('lambda');
% % ylabel('error');
% % axis([lambdas(1) lambdas(end) 1000 1500]);
% % hold off;

% **********************************
%             TEST 2
% **********************************
% K = 7;
% idxCV = splitGetCV(K, N);
% X = normalizedData(X);
% for k=1:1:K
%     [XTr, yTr, XTe, yTe] = splitGetTrTe(y, X, idxCV, k);
%     
%     tXTr = [ones(length(yTr),1) XTr];
%     tXTe = [ones(length(yTe),1) XTe];
%     
%     % mean err1_tr_ls = 1219
%     % mean err1_te_ls = 1250
%     % Matrix ill conditionned, so very bad error
% %      beta2_ls = leastSquares(yTr, tXTr);
% %      err2_tr_ls(k) = RMSE(yTr, tXTr*beta2_ls);
% %      err2_te_ls(k) = RMSE(yTe, tXTe*beta2_ls);
%     
%     % mean err1_tr_lsgd : 1219
%     % mean err1_te_lsgd : 1250
% %     alpha = 0.002875; %0.0029 without CV
% %     beta2_lsgd = leastSquaresGD(yTr, tXTr, alpha);
% %     err2_tr_lsgd(k) = RMSE(yTr, tXTr*beta2_lsgd);
% %     err2_te_lsgd(k) = RMSE(yTe, tXTe*beta2_lsgd);
% 
% %     lambdas = logspace(-5,5,10000);
% %     for i = 1:1:length(lambdas)
% %         lambda = lambdas(i);
% %         beta1_rr = ridgeRegression(yTr, tXTr, lambda);
% %         err1_tr_rr(k,i) = RMSE(yTr, tXTr*beta1_rr);
% %         err1_te_rr(k,i) = RMSE(yTe, tXTe*beta1_rr);
% %     end
% 
%     % mean err1_tr_rr : 1219
%     % mean err1_te_rr : 1250
%     lambda = 0.001;
%     beta2_rr = ridgeRegression(yTr, tXTr, lambda);
%     err2_tr_rr(k) = RMSE(yTr, tXTr*beta2_rr);
%     err2_te_rr(k) = RMSE(yTe, tXTe*beta2_rr);
% end
% 
% % figure
% % mseTr = mean(err1_tr_rr);
% % mseTe = mean(err1_te_rr);
% % 
% % [errStar, lambdaIdStar] = min(mseTe);
% % lambdaStar = lambdas(lambdaIdStar);
% %     
% % semilogx(lambdas, mseTr, 'b-o', lambdas, mseTe, 'r-x');
% % hold on;
% % legend('Train error', 'Test error', 'Location', 'southeast');
% % semilogx(lambdaStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
% % xlabel('lambda');
% % ylabel('error');
% % axis([lambdas(1) lambdas(end) 1000 1500]);
% % hold off;

% **********************************
%             TEST 3
% **********************************
% WARNING : Change preprocess fct to remove dummy encoding !
% Mean Tr : 121.76 + 148.72 + 164.95 = 435.43 (Improvement x3)
% Mean Te : 126.45 + 186.93 + 288 = 601.38 (Imrpvoement by x2)
% Cluster 1
% K = 5;
% N = length(y_cls1);
% idxCV = splitGetCV(K, N);
% 
% for k=1:1:K
%     [XTr, yTr, XTe, yTe] = splitGetTrTe(y_cls1, X_cls1, idxCV, k);
%     
%     tXTr = [ones(length(yTr),1) XTr];
%     tXTe = [ones(length(yTe),1) XTe];
%     
%     % mean err3_tr_ls_cls1 = 121.76
%     % mean err3_te_ls_clw1 = 126.45
%     % Matrix ill conditionned, so very bad error
% %       beta3_ls_cls1 = leastSquares(yTr, tXTr);
% %       err3_tr_ls_cls1(k) = RMSE(yTr, tXTr*beta3_ls_cls1);
% %       err3_te_ls_cls1(k) = RMSE(yTe, tXTe*beta3_ls_cls1);
% 
%     % mean err3_tr_lsgd_cls1 : 121.76
%     % mean err3_te_lsgd_cls1 : 126.45
% %      alpha = 0.01;
% %      beta3_lsgd_cls1 = leastSquaresGD(yTr, tXTr, alpha);
% %      err3_tr_lsgd_cls1(k) = RMSE(yTr, tXTr*beta3_lsgd_cls1);
% %      err3_te_lsgd_cls1(k) = RMSE(yTe, tXTe*beta3_lsgd_cls1);
% 
% %     lambdas = logspace(-5,5,10000);
% %     for i = 1:1:length(lambdas)
% %         lambda = lambdas(i);
% %         beta3_rr_cls1 = ridgeRegression(yTr, tXTr, lambda);
% %         err3_tr_rr_cls1(k,i) = RMSE(yTr, tXTr*beta3_rr_cls1);
% %         err3_te_rr_cls1(k,i) = RMSE(yTe, tXTe*beta3_rr_cls1);
% %     end
% 
%     % mean err3_tr_lsgd_cls1 : 121.78
%     % mean err3_te_lsgd_cls1 : 126.42
%     lambda = 15.8716;
%     beta3_rr_cls1 = ridgeRegression(yTr, tXTr, lambda);
%     err3_tr_rr_cls1(k) = RMSE(yTr, tXTr*beta3_rr_cls1);
%     err3_te_rr_cls1(k) = RMSE(yTe, tXTe*beta3_rr_cls1);
% end
% 
% % figure
% % mseTr = mean(err3_tr_rr_cls1);
% % mseTe = mean(err3_te_rr_cls1);
% % 
% % [errStar, lambdaIdStar] = min(mseTe);
% % lambdaStar = lambdas(lambdaIdStar);
% %     
% % semilogx(lambdas, mseTr, 'b-o', lambdas, mseTe, 'r-x');
% % hold on;
% % legend('Train error', 'Test error', 'Location', 'southeast');
% % semilogx(lambdaStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
% % xlabel('lambda');
% % ylabel('error');
% % axis([lambdas(1) lambdas(end) 0 200]);
% % hold off;

% Cluster2
% K = 2;
% N = length(y_cls2);
% idxCV = splitGetCV(K, N);
% 
% for k=1:1:K
%     [XTr, yTr, XTe, yTe] = splitGetTrTe(y_cls2, X_cls2, idxCV, k);
%     
%     tXTr = [ones(length(yTr),1) XTr];
%     tXTe = [ones(length(yTe),1) XTe];
%     
%     % mean err3_tr_lsgd_cls2 : 148.72
%     % mean err3_te_lsgd_cls2 : 186.93
%     % Matrix ill conditionned, so very bad error
% %     beta3_ls_cls2 = leastSquares(yTr, tXTr);
% %     err3_tr_ls_cls2(k) = RMSE(yTr, tXTr*beta3_ls_cls2);
% %     err3_te_ls_cls2(k) = RMSE(yTe, tXTe*beta3_ls_cls2);
% 
%     % mean err3_tr_lsgd_cls2 : 148.72
%     % mean err3_te_lsgd_cls2 : 186.93
% %      alpha = 0.01;
% %      beta3_lsgd_cls2 = leastSquaresGD(yTr, tXTr, alpha);
% %      err3_tr_lsgd_cls2(k) = RMSE(yTr, tXTr*beta3_lsgd_cls2);
% %      err3_te_lsgd_cls2(k) = RMSE(yTe, tXTe*beta3_lsgd_cls2);
% 
% %     lambdas = logspace(-5,5,10000);
% %     for i = 1:1:length(lambdas)
% %         lambda = lambdas(i);
% %         beta3_rr_cls2 = ridgeRegression(yTr, tXTr, lambda);
% %         err3_tr_rr_cls2(k,i) = RMSE(yTr, tXTr*beta3_rr_cls2);
% %         err3_te_rr_cls2(k,i) = RMSE(yTe, tXTe*beta3_rr_cls2);
% %     end
% 
% %     % mean err3_tr_lsgd_cls1 : 150.46
% %     % mean err3_te_lsgd_cls1 : 184.6226
%      lambda = 9.1537;
%      beta3_rr_cls2 = ridgeRegression(yTr, tXTr, lambda);
%      err3_tr_rr_cls2(k) = RMSE(yTr, tXTr*beta3_rr_cls2);
%      err3_te_rr_cls2(k) = RMSE(yTe, tXTe*beta3_rr_cls2);
% end
% 
% % figure
% % mseTr = mean(err3_tr_rr_cls2);
% % mseTe = mean(err3_te_rr_cls2);
% % 
% % [errStar, lambdaIdStar] = min(mseTe);
% % lambdaStar = lambdas(lambdaIdStar);
% %     
% % semilogx(lambdas, mseTr, 'b-o', lambdas, mseTe, 'r-x');
% % hold on;
% % legend('Train error', 'Test error', 'Location', 'southeast');
% % semilogx(lambdaStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
% % xlabel('lambda');
% % ylabel('error');
% % axis([lambdas(1) lambdas(end) 0 200]);
% % hold off;

% Cluster3
% K = 2;
% N = length(y_cls3);
% idxCV = splitGetCV(K, N);
% 
% for k=1:1:K
%     [XTr, yTr, XTe, yTe] = splitGetTrTe(y_cls3, X_cls3, idxCV, k);
%     
%     tXTr = [ones(length(yTr),1) XTr];
%     tXTe = [ones(length(yTe),1) XTe];
%     
%     % mean err3_tr_lsgd_cls3 : 164.95
%     % mean err3_te_lsgd_cls3 : 313.3477
%     % Matrix ill conditionned, so very bad error
% %     beta3_ls_cls3 = leastSquares(yTr, tXTr);
% %     err3_tr_ls_cls3(k) = RMSE(yTr, tXTr*beta3_ls_cls3);
% %     err3_te_ls_cls3(k) = RMSE(yTe, tXTe*beta3_ls_cls3);
% 
%     % mean err3_tr_lsgd_cls2 : 164.95
%     % mean err3_te_lsgd_cls2 : 313.35
% %      alpha = 0.01;
% %      beta3_lsgd_cls3 = leastSquaresGD(yTr, tXTr, alpha);
% %      err3_tr_lsgd_cls3(k) = RMSE(yTr, tXTr*beta3_lsgd_cls3);
% %      err3_te_lsgd_cls3(k) = RMSE(yTe, tXTe*beta3_lsgd_cls3);
% 
% %     lambdas = logspace(-5,5,10000);
% %     for i = 1:1:length(lambdas)
% %         lambda = lambdas(i);
% %         beta3_rr_cls3 = ridgeRegression(yTr, tXTr, lambda);
% %         err3_tr_rr_cls3(k,i) = RMSE(yTr, tXTr*beta3_rr_cls3);
% %         err3_te_rr_cls3(k,i) = RMSE(yTe, tXTe*beta3_rr_cls3);
% %     end
% 
% %     % mean err3_tr_lsgd_cls3 : 176.76
% %     % mean err3_te_lsgd_cls3 : 288.64
%      lambda = 12.1230;
%      beta3_rr_cls3 = ridgeRegression(yTr, tXTr, lambda);
%      err3_tr_rr_cls3(k) = RMSE(yTr, tXTr*beta3_rr_cls3);
%      err3_te_rr_cls3(k) = RMSE(yTe, tXTe*beta3_rr_cls3);
% end
% 
% % figure
% % mseTr = mean(err3_tr_rr_cls3);
% % mseTe = mean(err3_te_rr_cls3);
% % 
% % [errStar, lambdaIdStar] = min(mseTe);
% % lambdaStar = lambdas(lambdaIdStar);
% %     
% % semilogx(lambdas, mseTr, 'b-o', lambdas, mseTe, 'r-x');
% % hold on;
% % legend('Train error', 'Test error', 'Location', 'southeast');
% % semilogx(lambdaStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
% % xlabel('lambda');
% % ylabel('error');
% % axis([lambdas(1) lambdas(end) 0 350]);
% % hold off;

% **********************************
%             TEST 4
% **********************************
% WARNING : Change preprocess fct to remove dummy encoding !
% Mean Tr : 120.63 + 136.51 + 138.4 = 395.54 (Improvement)
% Mean Te : 126.85 + 202.07 + 355.89 = 685.44 (A?e !)
% Cluster 1
% K = 5;
% N = length(y_cls1);
% idxCV = splitGetCV(K, N);
% 
% for k=1:1:K
%     [XTr, yTr, XTe, yTe] = splitGetTrTe(y_cls1, X_cls1, idxCV, k);
%     
%     tXTr = [ones(length(yTr),1) XTr];
%     tXTe = [ones(length(yTe),1) XTe];
%     
%     % mean err3_tr_ls_cls1 = 120.40
%     % mean err3_te_ls_clw1 = 127.13
%     % Matrix ill conditionned, so very bad error
% %       beta4_ls_cls1 = leastSquares(yTr, tXTr);
% %       err4_tr_ls_cls1(k) = RMSE(yTr, tXTr*beta4_ls_cls1);
% %       err4_te_ls_cls1(k) = RMSE(yTe, tXTe*beta4_ls_cls1);
% 
%     % mean err3_tr_lsgd_cls1 : 120.40
%     % mean err3_te_lsgd_cls1 : 127.13
% %      alpha = 0.07;
% %      beta4_lsgd_cls1 = leastSquaresGD(yTr, tXTr, alpha);
% %      err4_tr_lsgd_cls1(k) = RMSE(yTr, tXTr*beta4_lsgd_cls1);
% %      err4_te_lsgd_cls1(k) = RMSE(yTe, tXTe*beta4_lsgd_cls1);
% 
% %     lambdas = logspace(-5,5,10000);
% %     for i = 1:1:length(lambdas)
% %         lambda = lambdas(i);
% %         beta4_rr_cls1 = ridgeRegression(yTr, tXTr, lambda);
% %         err4_tr_rr_cls1(k,i) = RMSE(yTr, tXTr*beta4_rr_cls1);
% %         err4_te_rr_cls1(k,i) = RMSE(yTe, tXTe*beta4_rr_cls1);
% %     end
% 
%     % mean err3_tr_lsgd_cls1 : 120.63
%     % mean err3_te_lsgd_cls1 : 126.85
%     lambda = 30.175;
%     beta4_rr_cls1 = ridgeRegression(yTr, tXTr, lambda);
%     err4_tr_rr_cls1(k) = RMSE(yTr, tXTr*beta4_rr_cls1);
%     err4_te_rr_cls1(k) = RMSE(yTe, tXTe*beta4_rr_cls1);
% end
% 
% % figure
% % mseTr = mean(err4_tr_rr_cls1);
% % mseTe = mean(err4_te_rr_cls1);
% % 
% % [errStar, lambdaIdStar] = min(mseTe);
% % lambdaStar = lambdas(lambdaIdStar);
% %     
% % semilogx(lambdas, mseTr, 'b-o', lambdas, mseTe, 'r-x');
% % hold on;
% % legend('Train error', 'Test error', 'Location', 'southeast');
% % semilogx(lambdaStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
% % xlabel('lambda');
% % ylabel('error');
% % axis([lambdas(1) lambdas(end) 0 200]);
% % hold off;

% Cluster2
% K = 2;
% N = length(y_cls2);
% idxCV = splitGetCV(K, N);
% 
% for k=1:1:K
%     [XTr, yTr, XTe, yTe] = splitGetTrTe(y_cls2, X_cls2, idxCV, k);
%     
%     tXTr = [ones(length(yTr),1) XTr];
%     tXTe = [ones(length(yTe),1) XTe];
%     
%     % mean err3_tr_lsgd_cls2 : 136.36
%     % mean err3_te_lsgd_cls2 : 205.73
%     % Matrix ill conditionned, so very bad error
% %     beta4_ls_cls2 = leastSquares(yTr, tXTr);
% %     err4_tr_ls_cls2(k) = RMSE(yTr, tXTr*beta4_ls_cls2);
% %     err4_te_ls_cls2(k) = RMSE(yTe, tXTe*beta4_ls_cls2);
% 
%     % mean err3_tr_lsgd_cls2 : 136.36
%     % mean err3_te_lsgd_cls2 : 205.73
% %      alpha = 0.3;
% %      beta4_lsgd_cls2 = leastSquaresGD(yTr, tXTr, alpha);
% %      err4_tr_lsgd_cls2(k) = RMSE(yTr, tXTr*beta4_lsgd_cls2);
% %      err4_te_lsgd_cls2(k) = RMSE(yTe, tXTe*beta4_lsgd_cls2);
% 
% %     lambdas = logspace(-5,5,10000);
% %     for i = 1:1:length(lambdas)
% %         lambda = lambdas(i);
% %         beta4_rr_cls2 = ridgeRegression(yTr, tXTr, lambda);
% %         err4_tr_rr_cls2(k,i) = RMSE(yTr, tXTr*beta4_rr_cls2);
% %         err4_te_rr_cls2(k,i) = RMSE(yTe, tXTe*beta4_rr_cls2);
% %     end
% 
%     % mean err3_tr_lsgd_cls1 : 136.51
%     % mean err3_te_lsgd_cls1 : 202.07
%      lambda = 0.5;
%      beta4_rr_cls2 = ridgeRegression(yTr, tXTr, lambda);
%      err4_tr_rr_cls2(k) = RMSE(yTr, tXTr*beta4_rr_cls2);
%      err4_te_rr_cls2(k) = RMSE(yTe, tXTe*beta4_rr_cls2);
% end
% 
% % figure
% % mseTr = mean(err4_tr_rr_cls2);
% % mseTe = mean(err4_te_rr_cls2);
% % 
% % [errStar, lambdaIdStar] = min(mseTe);
% % lambdaStar = lambdas(lambdaIdStar);
% %     
% % semilogx(lambdas, mseTr, 'b-o', lambdas, mseTe, 'r-x');
% % hold on;
% % legend('Train error', 'Test error', 'Location', 'southeast');
% % semilogx(lambdaStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
% % xlabel('lambda');
% % ylabel('error');
% % axis([lambdas(1) lambdas(end) 0 300]);
% % hold off;

% Cluster3
K = 2;
N = length(y_cls3);
idxCV = splitGetCV(K, N);

for k=1:1:K
    [XTr, yTr, XTe, yTe] = splitGetTrTe(y_cls3, X_cls3, idxCV, k);
    
    tXTr = [ones(length(yTr),1) XTr];
    tXTe = [ones(length(yTe),1) XTe];
    
    % mean err3_tr_lsgd_cls3 : 137.377
    % mean err3_te_lsgd_cls3 : 376.38
    % Matrix ill conditionned, so very bad error
%     beta4_ls_cls3 = leastSquares(yTr, tXTr);
%     err4_tr_ls_cls3(k) = RMSE(yTr, tXTr*beta4_ls_cls3);
%     err4_te_ls_cls3(k) = RMSE(yTe, tXTe*beta4_ls_cls3);

    % mean err3_tr_lsgd_cls2 : 137.38
    % mean err3_te_lsgd_cls2 : 376.38
%      alpha = 0.2;
%      beta4_lsgd_cls3 = leastSquaresGD(yTr, tXTr, alpha);
%      err4_tr_lsgd_cls3(k) = RMSE(yTr, tXTr*beta4_lsgd_cls3);
%      err4_te_lsgd_cls3(k) = RMSE(yTe, tXTe*beta4_lsgd_cls3);

%     lambdas = logspace(-5,5,10000);
%     for i = 1:1:length(lambdas)
%         lambda = lambdas(i);
%         beta4_rr_cls3 = ridgeRegression(yTr, tXTr, lambda);
%         err4_tr_rr_cls3(k,i) = RMSE(yTr, tXTr*beta4_rr_cls3);
%         err4_te_rr_cls3(k,i) = RMSE(yTe, tXTe*beta4_rr_cls3);
%     end

    % mean err3_tr_lsgd_cls3 : 138.4
    % mean err3_te_lsgd_cls3 : 355.89
     lambda = 0.5;
     beta4_rr_cls3 = ridgeRegression(yTr, tXTr, lambda);
     err4_tr_rr_cls3(k) = RMSE(yTr, tXTr*beta4_rr_cls3);
     err4_te_rr_cls3(k) = RMSE(yTe, tXTe*beta4_rr_cls3);
end

% figure
% mseTr = mean(err4_tr_rr_cls3);
% mseTe = mean(err4_te_rr_cls3);
% 
% [errStar, lambdaIdStar] = min(mseTe);
% lambdaStar = lambdas(lambdaIdStar);
%     
% semilogx(lambdas, mseTr, 'b-o', lambdas, mseTe, 'r-x');
% hold on;
% legend('Train error', 'Test error', 'Location', 'southeast');
% semilogx(lambdaStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
% xlabel('lambda');
% ylabel('error');
% axis([lambdas(1) lambdas(end) 0 400]);
% hold off;
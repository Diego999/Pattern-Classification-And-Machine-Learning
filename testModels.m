% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;
close all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;
N = length(y);

% 0) Constant baseline
% 0b) Constant baseline with clusters
% 1) Without any preprocess (no dummy encoding, no cluster separation, no normalization)
% 2) Same as 1) but with normalization
% 3) Separation with the cluster and normalization per cluster
% 4) Same as 3 but with dummy encoding
% 5) Same as 3 but with Polynomial basis
% 6) Same as 4 but with Polynomial basis

numberOfExperiments = 100;
proportionOfTraining = 0.8;

% %%
% % % **********************************
% % %             TEST 0
% % % **********************************
% % 
% % for i = 1:numberOfExperiments
% %     setSeed(28111993*i);
% %     [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
% %     
% %     % Nothing to learn
% %     beta = mean(yTr);
% %     
% %     err0(i) = RMSE(yTe, beta);
% % end
% % 
% % saveFile(err0, 'err0');
% % 
% % %%
% % % **********************************
% % %             TEST 0b
% % % **********************************
% % 
% % for i = 1:numberOfExperiments
% %     setSeed(28111993*i);
% %     [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
% %     
% %     [idx_cls1, idx_cls2, idx_cls3] = findClusters(yTr, XTr);
% %     yTr_cls1 = yTr(idx_cls1,:);
% %     yTr_cls2 = yTr(idx_cls2,:);
% %     yTr_cls3 = yTr(idx_cls3,:);
% %     
% %     [idx_cls1, idx_cls2, idx_cls3] = findClusters(yTe, XTe);
% %     yTe_cls1 = yTe(idx_cls1,:);
% %     yTe_cls2 = yTe(idx_cls2,:);
% %     yTe_cls3 = yTe(idx_cls3,:);
% %     
% %     % Nothing to learn
% %     beta_cls1 = mean(yTr_cls1);
% %     beta_cls2 = mean(yTr_cls2);
% %     beta_cls3 = mean(yTr_cls3);
% %     
% %     err0b_cls1(i) = RMSE(yTe_cls1, beta_cls1);
% %     err0b_cls2(i) = RMSE(yTe_cls2, beta_cls2);
% %     err0b_cls3(i) = RMSE(yTe_cls3, beta_cls3);
% % end
% % 
% % saveFile(err0b_cls1, 'err0b_cls1');
% % saveFile(err0b_cls2, 'err0b_cls2');
% % saveFile(err0b_cls3, 'err0b_cls3');
% % 
% % %%
% % % **********************************
% % %             TEST 1
% % % **********************************
% % 
% % K = 10;
% % 
% % for i = 1:1:numberOfExperiments
% %     setSeed(28111993*i);
% %     [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
% %     
% %     % Learning
% %     lambda = findLambda(K, yTr, XTr, 0, 1);
% %     
% %     tXTr = [ones(length(yTr),1) XTr];
% %     beta = ridgeRegression(yTr, tXTr, lambda);
% %     
% %     tXTe = [ones(length(yTe),1) XTe];
% %     err1(i) = RMSE(yTe, tXTe*beta);
% % end
% % 
% % saveFile(err1, 'err1');
% % 
% % %%
% % % **********************************
% % %             TEST 2
% % % **********************************
% % 
% % K = 10;
% % 
% % for i = 1:1:numberOfExperiments
% %     setSeed(28111993*i);
% %     [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, normalizedData(X));
% %     
% %     % Learning
% %     lambda = findLambda(K, yTr, XTr, 0, 1);
% %     
% %     tXTr = [ones(length(yTr),1) XTr];
% %     beta = ridgeRegression(yTr, tXTr, lambda);
% %     
% %     tXTe = [ones(length(yTe),1) XTe];
% %     err2(i) = RMSE(yTe, tXTe*beta);
% % end
% % 
% % saveFile(err2, 'err2');
% 
% %%
% % **********************************
% %             TEST 3
% % **********************************
% % Don't forget to modify process in order to use only normalization !
% 
% k1 = 5;
% k2 = 6;
% k3 = 4;
% 
% for i = 1:1:numberOfExperiments
%     setSeed(28111993*i);
%     [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
%     
%     [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTr, XTr);
% 
%     % Learning
%      lambda_cls1 = findLambda(k1, y_cls1, X_cls1, 0, 1);
%      lambda_cls2 = findLambda(k2, y_cls2, X_cls2, 0, 1);
%      lambda_cls3 = findLambda(k3, y_cls3, X_cls3, 0, 1);
%     
%      tX_cls1 = [ones(length(y_cls1),1) X_cls1];
%      tX_cls2 = [ones(length(y_cls2),1) X_cls2];
%      tX_cls3 = [ones(length(y_cls3),1) X_cls3];
%     
%      beta_cls1 = ridgeRegression(y_cls1, tX_cls1, lambda_cls1);
%      beta_cls2 = ridgeRegression(y_cls2, tX_cls2, lambda_cls2);
%      beta_cls3 = ridgeRegression(y_cls3, tX_cls3, lambda_cls3);
%      
%      [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTe, XTe);
%  
%      if size(X_cls1,2) ~= size(X_cls2,2) || size(X_cls2,2) ~= size(X_cls3,2)
%          continue
%      end
%      
%      tXTe_cls1 = [ones(length(y_cls1),1) X_cls1];
%      tXTe_cls2 = [ones(length(y_cls2),1) X_cls2];
%      tXTe_cls3 = [ones(length(y_cls3),1) X_cls3];
%      
%      err3_cls1(i) = RMSE(y_cls1, tXTe_cls1*beta_cls1);
%      err3_cls2(i) = RMSE(y_cls2, tXTe_cls2*beta_cls2);
%      err3_cls3(i) = RMSE(y_cls3, tXTe_cls3*beta_cls3);
% end
% 
% saveFile(err3_cls1, 'err3_cls1');
% saveFile(err3_cls2, 'err3_cls2');
% saveFile(err3_cls3, 'err3_cls3');
% 
% %%
% % **********************************
% %             TEST 4
% % **********************************
% 
% k1 = 5;
% k2 = 6;
% k3 = 4;
% 
% for i = 1:1:numberOfExperiments
%     setSeed(28111993*i);
%     [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
%     
%     [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTr, XTr);
% 
%     % Learning
%      lambda_cls1 = findLambda(k1, y_cls1, X_cls1, 0, 1);
%      lambda_cls2 = findLambda(k2, y_cls2, X_cls2, 0, 1);
%      lambda_cls3 = findLambda(k3, y_cls3, X_cls3, 0, 1);
%     
%      tX_cls1 = [ones(length(y_cls1),1) X_cls1];
%      tX_cls2 = [ones(length(y_cls2),1) X_cls2];
%      tX_cls3 = [ones(length(y_cls3),1) X_cls3];
%     
%      beta_cls1 = ridgeRegression(y_cls1, tX_cls1, lambda_cls1);
%      beta_cls2 = ridgeRegression(y_cls2, tX_cls2, lambda_cls2);
%      beta_cls3 = ridgeRegression(y_cls3, tX_cls3, lambda_cls3);
%      
%      [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTe, XTe);
%  
%      tXTe_cls1 = [ones(size(y_cls1,1),1) X_cls1];
%      tXTe_cls2 = [ones(size(y_cls2,1),1) X_cls2];
%      tXTe_cls3 = [ones(size(y_cls3,1),1) X_cls3];
%      
%      err4_cls1(i) = RMSE(y_cls1, tXTe_cls1*beta_cls1);
%      err4_cls2(i) = RMSE(y_cls2, tXTe_cls2*beta_cls2);
%      err4_cls3(i) = RMSE(y_cls3, tXTe_cls3*beta_cls3);
% 
% end
% 
% saveFile(err4_cls1, 'err4_cls1');
% saveFile(err4_cls2, 'err4_cls2');
% saveFile(err4_cls3, 'err4_cls3');

%%
% **********************************
%             TEST 5
% **********************************
% Don't forget to modify process in order to use only normalization !

k1 = 5;
k2 = 6;
k3 = 4;

d1 = 3;
d2 = 5;
d3 = 2;

for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
    
    % Learning
     lambda_cls1 = findLambda(k1, yTr, XTr, 1, d1, d2, d3);
     lambda_cls2 = findLambda(k2, yTr, XTr, 2, d1, d2, d3);
     lambda_cls3 = findLambda(k3, yTr, XTr, 3, d1, d2, d3);
    
     [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTr, XTr, d1, d2, d3);

     tX_cls1 = [ones(length(y_cls1),1) X_cls1];
     tX_cls2 = [ones(length(y_cls2),1) X_cls2];
     tX_cls3 = [ones(length(y_cls3),1) X_cls3];
    
     beta_cls1 = ridgeRegression(y_cls1, tX_cls1, lambda_cls1);
     beta_cls2 = ridgeRegression(y_cls2, tX_cls2, lambda_cls2);
     beta_cls3 = ridgeRegression(y_cls3, tX_cls3, lambda_cls3);
     
     [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTe, XTe, d1, d2, d3);
 
     if size(X_cls1,2) ~= size(X_cls2,2) || size(X_cls2,2) ~= size(X_cls3,2)
         continue
     end
     
     tXTe_cls1 = [ones(length(y_cls1),1) X_cls1];
     tXTe_cls2 = [ones(length(y_cls2),1) X_cls2];
     tXTe_cls3 = [ones(length(y_cls3),1) X_cls3];
     
     err5_cls1(i) = RMSE(y_cls1, tXTe_cls1*beta_cls1);
     err5_cls2(i) = RMSE(y_cls2, tXTe_cls2*beta_cls2);
     err5_cls3(i) = RMSE(y_cls3, tXTe_cls3*beta_cls3);
end

saveFile(err5_cls1, 'err5_cls1');
saveFile(err5_cls2, 'err5_cls2');
saveFile(err5_cls3, 'err5_cls3');

%%
s = [1 numberOfExperiments];

err0 = openFile('err0', s);
err0b = openFile('err0b_cls1', s) + openFile('err0b_cls2',s) + openFile('err0b_cls3',s);
err1 = openFile('err1', s);
err2 = openFile('err2', s);
err3 = openFile('err3_cls1', s) + openFile('err3_cls2',s) + openFile('err3_cls3',s);
err4 = openFile('err4_cls1', s) + openFile('err4_cls2',s) + openFile('err4_cls3',s);

figure;
boxplot([err0' err0b' err1' err2' err3' err4']);
legend(findobj(gca,'Tag','Box'),'0 Constant model','0b Constant model per cluster', '1 Least squares', '2 + Normalization', '3 + per cluster', '4 + Dummy encoding');
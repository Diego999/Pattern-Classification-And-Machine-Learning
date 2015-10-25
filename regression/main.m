% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;
close all;

load('Sydney_regression.mat');

y = y_train;
X = X_train;
N = length(y);

% 1) Constant baseline
% 2) Without any preprocess (no dummy encoding, no cluster separation, no normalization)
% 3) Same as 2) but with normalization
% 4) Constant baseline with clusters
% 5) Separation with the cluster and normalization per cluster
% 6) Same as 5 but with dummy encoding
% 7) Same as 5 but with Polynomial basis
% 8) Same as 6 but with Polynomial basis

numberOfExperiments = 30;
proportionOfTraining = 0.8;

%%
% **********************************
%             TEST 1
% **********************************

for i = 1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
    
    % Nothing to learn
    beta = mean(yTr);
    
    err1(i) = RMSE(yTe, beta);
end

saveFile(err1, 'results/err1');

%%
% **********************************
%             TEST 2
% **********************************
% Remove in preprocessCluster around line 68 the normalization 
% Remove in preprocessCluster around line 75 the dummy encoding
K = 10;

for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    fprintf('%d\n', i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
    
    % Learning
    lambda = findLambda(K, yTr, XTr, 0, 1);
    
    tXTr = [ones(length(yTr),1) XTr];
    beta = ridgeRegression(yTr, tXTr, lambda);
    
    tXTe = [ones(length(yTe),1) XTe];
    err2(i) = RMSE(yTe, tXTe*beta);
end

saveFile(err2, 'results/err2');

%%
% **********************************
%             TEST 3
% **********************************
% Add in preprocessCluster around line 68 the normalization 
% Remove in preprocessCluster around line 75 the dummy encoding

K = 10;

for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, normalizedData(X));

    % Learning
    lambda = findLambda(K, yTr, XTr, 0);
    
    tXTr = [ones(length(yTr),1) XTr];
    beta = ridgeRegression(yTr, tXTr, lambda);
    
    tXTe = [ones(length(yTe),1) XTe];
    err3(i) = RMSE(yTe, tXTe*beta);
end

saveFile(err3, 'results/err3');

%%
% **********************************
%             TEST 4
% **********************************

for i = 1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
    
    [idx_cls1, idx_cls2, idx_cls3] = findClusters(yTr, XTr);
    yTr_cls1 = yTr(idx_cls1,:);
    yTr_cls2 = yTr(idx_cls2,:);
    yTr_cls3 = yTr(idx_cls3,:);
    
    [idx_cls1, idx_cls2, idx_cls3] = findClusters(yTe, XTe);
    yTe_cls1 = yTe(idx_cls1,:);
    yTe_cls2 = yTe(idx_cls2,:);
    yTe_cls3 = yTe(idx_cls3,:);
    
    % Nothing to learn
    beta_cls1 = mean(yTr_cls1);
    beta_cls2 = mean(yTr_cls2);
    beta_cls3 = mean(yTr_cls3);
    
    err4_cls1(i) = RMSE(yTe_cls1, beta_cls1);
    err4_cls2(i) = RMSE(yTe_cls2, beta_cls2);
    err4_cls3(i) = RMSE(yTe_cls3, beta_cls3);
end

saveFile(err4_cls1, 'results/err4_cls1');
saveFile(err4_cls2, 'results/err4_cls2');
saveFile(err4_cls3, 'results/err4_cls3');

%%
% **********************************
%             TEST 5
% **********************************
% Add in preprocessCluster around line 68 the normalization 
% Remove in preprocessCluster around line 75 the dummy encoding

k1 = 10;
k2 = 8;
k3 = 6;

for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);

    [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTr, XTr);

    % Learning
     lambda_cls1 = findLambda(k1, y_cls1, X_cls1, 0, 1);
     lambda_cls2 = findLambda(k2, y_cls2, X_cls2, 0, 1);
     lambda_cls3 = findLambda(k3, y_cls3, X_cls3, 0, 1);
    
     tX_cls1 = [ones(length(y_cls1),1) X_cls1];
     tX_cls2 = [ones(length(y_cls2),1) X_cls2];
     tX_cls3 = [ones(length(y_cls3),1) X_cls3];
    
     beta_cls1 = ridgeRegression(y_cls1, tX_cls1, lambda_cls1);
     beta_cls2 = ridgeRegression(y_cls2, tX_cls2, lambda_cls2);
     beta_cls3 = ridgeRegression(y_cls3, tX_cls3, lambda_cls3);
     
     [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTe, XTe);
 
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

saveFile(err5_cls1, 'results/err5_cls1');
saveFile(err5_cls2, 'results/err5_cls2');
saveFile(err5_cls3, 'results/err5_cls3');

%%
% **********************************
%             TEST 6
% **********************************
% Add in preprocessCluster around line 68 the normalization 
% Add in preprocessCluster around line 75 the dummy encoding

k1 = 10;
k2 = 8;
k3 = 6;

for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
      
    [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTr, XTr);

    % Learning
     lambda_cls1 = findLambda(k1, y_cls1, X_cls1, 0, 1);
     lambda_cls2 = findLambda(k2, y_cls2, X_cls2, 0, 1);
     lambda_cls3 = findLambda(k3, y_cls3, X_cls3, 0, 1);
    
     tX_cls1 = [ones(length(y_cls1),1) X_cls1];
     tX_cls2 = [ones(length(y_cls2),1) X_cls2];
     tX_cls3 = [ones(length(y_cls3),1) X_cls3];
    
     beta_cls1 = ridgeRegression(y_cls1, tX_cls1, lambda_cls1);
     beta_cls2 = ridgeRegression(y_cls2, tX_cls2, lambda_cls2);
     beta_cls3 = ridgeRegression(y_cls3, tX_cls3, lambda_cls3);
     
     [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTe, XTe);
 
     tXTe_cls1 = [ones(size(y_cls1,1),1) X_cls1];
     tXTe_cls2 = [ones(size(y_cls2,1),1) X_cls2];
     tXTe_cls3 = [ones(size(y_cls3,1),1) X_cls3];
     
     err6_cls1(i) = RMSE(y_cls1, tXTe_cls1*beta_cls1);
     err6_cls2(i) = RMSE(y_cls2, tXTe_cls2*beta_cls2);
     err6_cls3(i) = RMSE(y_cls3, tXTe_cls3*beta_cls3);

end

saveFile(err6_cls1, 'results/err6_cls1');
saveFile(err6_cls2, 'results/err6_cls2');
saveFile(err6_cls3, 'results/err6_cls3');

%%
% **********************************
%             TEST 7
% **********************************
% Add in preprocessCluster around line 68 the normalization 
% Remove in preprocessCluster around line 75 the dummy encoding

k1 = 10;
k2 = 8;
k3 = 6;

d1 = 3;
d2 = 6;
d3 = 3;

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
 
     tXTe_cls1 = [ones(length(y_cls1),1) X_cls1];
     tXTe_cls2 = [ones(length(y_cls2),1) X_cls2];
     tXTe_cls3 = [ones(length(y_cls3),1) X_cls3];
     
     err7_cls1(i) = RMSE(y_cls1, tXTe_cls1*beta_cls1);
     err7_cls2(i) = RMSE(y_cls2, tXTe_cls2*beta_cls2);
     err7_cls3(i) = RMSE(y_cls3, tXTe_cls3*beta_cls3);
end

saveFile(err7_cls1, 'results/err7_cls1');
saveFile(err7_cls2, 'results/err7_cls2');
saveFile(err7_cls3, 'results/err7_cls3');

%%
% **********************************
%             TEST 8
% **********************************
% Add in preprocessCluster around line 68 the normalization 
% Add in preprocessCluster around line 75 the dummy encoding

k1 = 10;
k2 = 8;
k3 = 6;

d1 = 3;
d2 = 5;
d3 = 2;

for i = 1:1:numberOfExperiments
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
    fprintf('%d\n', i); 
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
 
     tXTe_cls1 = [ones(length(y_cls1),1) X_cls1];
     tXTe_cls2 = [ones(length(y_cls2),1) X_cls2];
     tXTe_cls3 = [ones(length(y_cls3),1) X_cls3];
     
     err8_cls1(i) = RMSE(y_cls1, tXTe_cls1*beta_cls1);
     err8_cls2(i) = RMSE(y_cls2, tXTe_cls2*beta_cls2);
     err8_cls3(i) = RMSE(y_cls3, tXTe_cls3*beta_cls3);
end

saveFile(err8_cls1, 'results/err8_cls1');
saveFile(err8_cls2, 'results/err8_cls2');
saveFile(err8_cls3, 'results/err8_cls3');
%%
s = [1 numberOfExperiments];

[y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(y, X);

N = length(y_cls1) + length(y_cls2) + length(y_cls3);
ratio_cls1 = size(X_cls1, 1)/N;
ratio_cls2 = size(X_cls2, 1)/N;
ratio_cls3 = size(X_cls3, 1)/N;

err1 = openFile('results/err1', s);
err2 = openFile('results/err2',s);
err3 = openFile('results/err3', s);

err4 = sqrt((1/N)*( ...
            (size(X_cls1, 1)*(openFile('results/err4_cls1', s).^2)) ...
          + (size(X_cls2, 1)*(openFile('results/err4_cls2', s).^2)) ...
          + (size(X_cls3, 1)*(openFile('results/err4_cls3', s).^2)) ...
     ));
     
 
err5 = sqrt((1/N)*( ...
            (size(X_cls1, 1)*(openFile('results/err5_cls1', s).^2)) ...
          + (size(X_cls2, 1)*(openFile('results/err5_cls2', s).^2)) ...
          + (size(X_cls3, 1)*(openFile('results/err5_cls3', s).^2)) ...
     ));
 
err6 = sqrt((1/N)*( ...
        (size(X_cls1, 1)*(openFile('results/err6_cls1', s).^2)) ...
      + (size(X_cls2, 1)*(openFile('results/err6_cls2', s).^2)) ...
      + (size(X_cls3, 1)*(openFile('results/err6_cls3', s).^2)) ...
 ));

err7 = sqrt((1/N)*( ...
        (size(X_cls1, 1)*(openFile('results/err7_cls1', s).^2)) ...
      + (size(X_cls2, 1)*(openFile('results/err7_cls2', s).^2)) ...
      + (size(X_cls3, 1)*(openFile('results/err7_cls3', s).^2)) ...
 ));

err8 = sqrt((1/N)*( ...
        (size(X_cls1, 1)*(openFile('results/err8_cls1', s).^2)) ...
      + (size(X_cls2, 1)*(openFile('results/err8_cls2', s).^2)) ...
      + (size(X_cls3, 1)*(openFile('results/err8_cls3', s).^2)) ...
 ));

 
% 1) Constant baseline
% 2) Without any preprocess (no dummy encoding, no cluster separation, no normalization)
% 3) Same as 2) but with normalization
% 4) Constant baseline with clusters
% 5) Separation with the cluster and normalization per cluster
% 6) Same as 5 but with dummy encoding
% 7) Same as 5 but with Polynomial basis
% 8) Same as 6 but with Polynomial basis

figure;
boxplot([err1' err2' err3' err4' err5' err6' err7' err8']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 Constant model', ...
'2 Ridge regression', ...
'3 + Normalization', ...
'4 Constant model per cluster', ...
'5 Ridge regression per cluster with normalization', ...
'6 + Dummy encoding', ...
'7 Same as 5 + polynomial basis', ...
'8 Same as 6 + polynomial basis');
set(h_legend,'FontSize',12);
set(gca, 'XGrid','on')
set(gca, 'YGrid','on')
ylim([0 3000])
set(gca,'YTick',0:100:3000)
xlabel('Model');
ylabel('RMSE');
print('report/figures/models','-djpeg','-noui')
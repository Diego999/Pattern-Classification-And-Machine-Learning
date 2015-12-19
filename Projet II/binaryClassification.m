%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

numberOfExperiments = 10;
proportionOfTraining = 0.8;
K = 10;
M = 150;

train.y(find(train.y == 2)) = 1;
train.y(find(train.y == 3)) = 1;
train.y(find(train.y == 4)) = 2;

%% Create data
fprintf('Creating Train & Test sets\n');
tic
[Tr, Te] = createTrainingTestingCNN(train.X_cnn, train.y, proportionOfTraining);
toc

%% Prepare data

fprintf('Prepare the data for the training\n');
tic
[Tr, Te] = prepareDataCNN(Tr, Te, M);
toc

%%
% 50%
%**********************************
%            TEST 1
%**********************************

fprintf('Test 1\n')
for i = 1:numberOfExperiments
    fprintf('%d ', i);
    setSeed(28111993*i);
    out = 1;
    err1(i) = balancedErrorRate(Te.y, repmat([out], length(Tr.y), 1));
end
fprintf('\n%f\n', mean(err1));
saveFile(err1, 'results/binary/err1');

%% 9.96%
%**********************************
%            TEST 2
%**********************************

N = length(Tr.y);

% Setup NN.
inputSize  = M;
innerSize  = 1000;
numepochs = 20;
batchsize = 100;
learningRate = 3;
binaryClassification = true;

Tr_ = Tr;
Te_ = Te;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
        
    idxCV = splitGetCV(K, length(TTr.y));
    
    % K-fold
    for k=1:1:K
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);
        
        [errnZ, nnPrednZ] = neuralNetworks(TTTr.nZ, TTTr.y, TTTe.nZ, TTTe.y, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

        err_te(k) = errnZ;
    end
    err2(j) = mean(err_te);
end
fprintf('\n%f\n', mean(err2));
saveFile(err2, 'results/binary/err2');

%% 9.97%
%**********************************
%            TEST 3
%**********************************

N = length(Tr.y);

% Setup 
NLeaves = 200;
Tr_ = Tr;
Te_ = Te;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
        
    idxCV = splitGetCV(K, length(TTr.y));
    tic
    % K-fold
    for k=1:1:K
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);
        
        CMdl = fitensemble(TTTr.Z, TTTr.y, 'Bag', NLeaves, 'Tree', 'Type', 'classification');
        yhat = predict(CMdl, TTTe.Z);

        err_te(k) = balancedErrorRate(TTTe.y, yhat);
    end
    toc
    err3(j) = mean(err_te);
end
fprintf('\n%f\n', mean(err3));
saveFile(err3, 'results/binary/err3');

%% 9.98%
%**********************************
%            TEST 4
%**********************************

N = length(Tr.y);

% Setup 
NTrees = 400;

Tr_ = Tr;
Te_ = Te;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
        
    idxCV = splitGetCV(K, length(TTr.y));
    
    % K-fold
    for k=1:1:K
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);
        
        BaggedEnsemble = TreeBagger(NTrees, TTTr.Z, TTTr.y);
        yhat = str2double(predict(BaggedEnsemble, TTTe.Z));

        err_te(k) = balancedErrorRate(TTTe.y, yhat);
    end
    err4(j) = mean(err_te);
end
fprintf('\n%f\n', mean(err4));
saveFile(err4, 'results/binary/err4');

%% 8.59%
%**********************************
%            TEST 5
%**********************************

N = length(Tr.y);

% Setup 
nbWeak = 1024;
maxDepth = 4;
pBoost=struct('nWeak',nbWeak,'pTree',struct('maxDepth',maxDepth));

Tr_ = Tr;
Te_ = Te;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
        
    idxCV = splitGetCV(K, length(TTr.y));
    
    % K-fold
    for k=1:1:K
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);

        model = adaBoostTrain(TTTr.nZ(TTTr.y==1, :), TTTr.nZ(TTTr.y==2,:), pBoost);

        yp = TTTe.y(find(TTTe.y == 1));
        yn = TTTe.y(find(TTTe.y == 2));

        fp = adaBoostApply(TTTe.nZ(TTTe.y==1, :), model);
        fp = double(fp > 0);
        fp = fp + ones(length(fp), 1);
        fn = adaBoostApply(TTTe.nZ(TTTe.y==2, :), model);
        fn = double(fn > 0);
        fn = fn + ones(length(fn), 1);

        yhat = [fp; fn];
        ytrue = [yp; yn];

        err_te(k) = balancedErrorRate(ytrue, yhat);
    end
    err5(j) = mean(err_te);
end
fprintf('\n%f\n', mean(err5));
saveFile(err5, 'results/binary/err5');

%% Plot
s = [1 numberOfExperiments];

err1 = openFile('results/binary/err1', s);
err2 = openFile('results/binary/err2',s);
err3 = openFile('results/binary/err3', s);
err4 = openFile('results/binary/err4', s);
err5 = openFile('results/binary/err5', s);

figure;
boxplot([err1' err2' err3' err4' err5']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 ', ...
'2 ', ...
'3 ', ...
'4 ', ...
'5 ');
set(gca, 'XGrid','on')
set(gca, 'YGrid','on')
set(gca,'LineWidth',1.5);
ylim([0 0.55])
set(gca,'YTick',0:0.05:0.55)
xlabel('Model');
ylabel('BER');
%print('../report/figures/models','-djpeg','-noui')
%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

numberOfExperiments = 10;
proportionOfTraining = 0.8;
K = 10;
M = 1000;

train.y(find(train.y == 2)) = 1;
train.y(find(train.y == 3)) = 1;
train.y(find(train.y == 4)) = 2;

%% Create data
fprintf('Creating Train & Test sets\n');
tic
[Tr, Te] = createTrainingTestingHOG(train.X_hog, train.y, 1.0);
toc

%% Prepare data

fprintf('Prepare the data for the training\n');
tic
[Tr, Te] = prepareDataHOG(Tr, Te, M);
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
    err1(i) = 0.5;%balancedErrorRate(Te.y, repmat([out], length(Tr.y), 1));
end
fprintf('\n%f\n', mean(err1));
saveFile(err1, 'results/binary/err1_');

%% 23.68%
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

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
   
    [errnZ, nnPrednZ] = neuralNetworks(TTr.nZ, TTr.y, TTe.nZ, TTe.y, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

    err2(j) = errnZ;
end
fprintf('\n%f\n', mean(err2));
saveFile(err2, 'results/binary/err2_');

%% 0.4 %
%**********************************
%            TEST 3
%**********************************

N = length(Tr.y);

% Setup 
NLeaves = 200;
Tr_ = Tr;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
     
    CMdl = fitensemble(TTr.Z, TTr.y, 'Bag', NLeaves, 'Tree', 'Type', 'classification');
    yhat = predict(CMdl, TTe.Z);

    err3(j) = balancedErrorRate(TTe.y, yhat);
end
fprintf('\n%f\n', mean(err3));
saveFile(err3, 'results/binary/err3_');

%% %
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
        
    BaggedEnsemble = TreeBagger(NTrees, TTr.Z, TTr.y);
    yhat = str2double(predict(BaggedEnsemble, TTe.Z));

    err4(j) = balancedErrorRate(TTe.y, yhat);
end
fprintf('\n%f\n', mean(err4));
saveFile(err4, 'results/binary/err4_');

%% %
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
saveFile(err5, 'results/binary/err5_');

%% Plot
s = [1 numberOfExperiments];

err1 = openFile('results/binary/err1_', s);
err2 = openFile('results/binary/err2_',s);
err3 = openFile('results/binary/err3_', s);
err4 = openFile('results/binary/err4_', s);
err5 = openFile('results/binary/err5_', s);

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
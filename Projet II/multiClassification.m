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
saveFile(err1, 'results/multi/err1');

%%
%**********************************
%            TEST 2
%**********************************

N = length(Tr.y);

% Setup NN.
inputSize  = M;
innerSize  = 700;
numepochs = 20;
batchsize = 100;
learningRate = 2;
binaryClassification = false;

Tr_ = Tr;
Te_ = Te;

for j = 1:1:numberOfExperiments
    setSeed(28111993*i);

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
saveFile(err2, 'results/multi/err2');

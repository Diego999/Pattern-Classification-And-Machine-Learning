%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;
load test/test.mat;

numberOfExperiments = 30;
proportionOfTraining = 0.8;
K = 10;
M = 150;

%% Create data
fprintf('Creating Train & Test sets\n');
tic
[Tr, Te] = createTrainingTestingCNN(train.X_cnn, train.y, 1.0);
[Te] = createTestingCNN(test.X_cnn);
toc

%% Prepare data

fprintf('Prepare the data for the training\n');
tic
[Tr, Te] = prepareDataCNN(Tr, Te, M);
[Te] = prepareDataCNNTesting(Te, M);
toc

%% COMPUTING ESTIMATE ERROR

[TTr, TTe] = splitProp(proportionOfTraining, Tr, false);

idxCV = splitGetCV(K, length(TTr.y));

% Setup NN.
inputSize  = M;
innerSize  = 700;
numepochs = 20;
batchsize = 100;
learningRate = 2;
binaryClassification = false;

% K-fold
for k=1:1:K
    fprintf('%d fold\n', k);
    [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);

    [errnZ, nnPrednZ] = neuralNetworks(TTTr.nZ, TTTr.y, TTTe.nZ, TTTe.y, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

    err(k) = errnZ;
end

% 0.085 +- 0.010

u = mean(err); 
o = std(err);

%% TRAINING FOR PREDICTION

% Setup NN.
inputSize  = M;
innerSize  = 700;
numepochs = 20;
batchsize = 100;
learningRate = 2;
binaryClassification = false;

% Train & test

[Ytest] = neuralNetworksGenPred(Tr.nZ, Tr.y, Te.nZ, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

% Write the prediction
save('pred_multiclass', 'Ytest');
%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;


numberOfExperiments = 30;
proportionOfTraining = 0.8;
K = 10;
M = 300;

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

%% Cross validate

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
    
if binaryClassification    
    Tr_.y(find(Tr_.y == 2)) = 1;
    Tr_.y(find(Tr_.y == 3)) = 1;
    Tr_.y(find(Tr_.y == 4)) = 2;
    
    Te_.y(find(Te_.y == 2)) = 1;
    Te_.y(find(Te_.y == 3)) = 1;
    Te_.y(find(Te_.y == 4)) = 2;
end

innerSizes = [500 600 700 800 900 100]; % Best so far 700 : 8.59%
learningRates = [2.0 3.0 4.0 5.0]; % Best so far 2 : 8.3%

for j = 1:1:numberOfExperiments
    setSeed(28111993*i);

    fprintf('%d : Split the data\n', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
    
    
    idxCV = splitGetCV(K, length(TTr.y));
    
    % K-fold
    for k=1:1:K
        fprintf('%d : %dth fold\n', j, k);
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);
        
        %for i = 1:1:length(innerSizes)
        %    innerSize = innerSizes(i);
        for i = 1:1:length(learningRates)
            learningRate = learningRates(i);

            [errnZ, nnPrednZ] = neuralNetworks(TTTr.nZ, TTTr.y, TTTe.nZ, TTTe.y, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

            err_te(k,j) = errnZ;
        end
        %end
    end

    mseTe = mean(err_te);

    %[errStar, innerSizeStar] = min(mseTe);
    %lambdaStar = innerSizes(innerSizeStar);
    
    [errStar, learningRateStar] = min(mseTe);
    learningRateStar = learningRates(learningRateStar);
end



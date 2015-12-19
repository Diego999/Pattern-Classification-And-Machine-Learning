%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;


numberOfExperiments = 1;
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

%% Cross validate

N = length(Tr.y);

%% NN
% Setup NN.
inputSize  = M;
innerSize  = 1000;
numepochs = 20;
batchsize = 100;
learningRate = 2;
binaryClassification = true;

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

innerSizes = [1000]; % Best so far 1000 : 9.35%
learningRates = [3.0]; % Best so far 3 : 9.56%

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

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

            err_te(k,i) = errnZ;
        end
        %end
    end

    mseTe = mean(err_te);

    %[errStar, innerSizeStar] = min(mseTe);
    %lambdaStar = innerSizes(innerSizeStar);
    
    [errStar, learningRateStar] = min(mseTe);
    learningRateStar = learningRates(learningRateStar);
end

%% DT

binaryClassification = true;

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


% 10.824 with 200
NLeaves = [200];

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d : Split the data\n', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
    
    idxCV = splitGetCV(K, length(TTr.y));
    
    % K-fold
    for k=1:1:K
        fprintf('%d : %dth fold\n', j, k);
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);
        
        for i = 1:1:length(NLeaves)
            NLeave = NLeaves(i);
            
            CMdl = fitensemble(TTTr.Z, TTTr.y, 'Bag', NLeave, 'Tree', 'Type', 'classification');
            yhat = predict(CMdl, TTTe.Z);

            err_te(k,i) = balancedErrorRate(TTTe.y, yhat);
        end
    end

    mseTe = mean(err_te);

    [errStar, NLeaveStarId] = min(mseTe);
    NLeaveStar = NLeaves(NLeaveStarId);
end

%% RF
    
binaryClassification = true;

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


% Best is 400 (9.92), however NTrees growsl linearly and so, it is a good
% comprises between speed an accuracy
NTrees = [400];

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d : Split the data\n', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
    
    idxCV = splitGetCV(K, length(TTr.y));
    
    % K-fold
    for k=1:1:K
        fprintf('%d : %dth fold\n', j, k);
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);
        
        for i = 1:1:length(NTrees)
            NTree = NTrees(i);
            tic
            BaggedEnsemble = TreeBagger(NTree, TTTr.Z, TTTr.y);
            yhat = str2double(predict(BaggedEnsemble, TTTe.Z));
            toc
            err_te(k,i) = balancedErrorRate(TTTe.y, yhat);
        end
    end

    mseTe = mean(err_te);

    [errStar, NTreeStarId] = min(mseTe);
    NTreeStar = NTrees(NTreeStarId);
end

%% BT

binaryClassification = true;

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

nWeak = 1024;
maxDepth = 4;
nWeaks = [1024]; % 9.06 for 1024
maxDepths = [4 8 16 32 64]; % 9.06 for 4

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d : Split the data\n', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
    
    idxCV = splitGetCV(K, length(TTr.y));
    
    % K-fold
    for k=1:1:K
        fprintf('%d : %dth fold\n', j, k);
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);
        
        %for i = 1:1:length(nWeaks)
        %    nWeak = nWeaks(i);
        for i = 1:1:length(maxDepths)
            maxDepth = maxDepths(i);

            pBoost=struct('nWeak',nWeak,'pTree',struct('maxDepth',maxDepth));
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

            err_te(k,i) = balancedErrorRate(ytrue, yhat);
            fprintf('%f\n', err_te(k,i));
        %end
        end
    end

    mseTe = mean(err_te);

    %[errStar, nbWeaksStarId] = min(mseTe);
    %nbWeaksStar = nWeaks(nbWeaksStarId);
    
    [errStar, maxDepthStarId] = min(mseTe);
    maxDepthStar = maxDepths(maxDepthStarId);
end
% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

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
[Tr, Te] = createTrainingTestingCNN(train.X_cnn, train.y, 1.0);
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

Tr_ = Tr;
Te_ = Te;

% 10.85 with 500
NLeaves = [100 200 300 400 500 600 700 800 900];

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
    
Tr_ = Tr;
Te_ = Te;

% Best is 500 (0.1072), however NTrees growsl linearly and so, it is a good
% comprises between speed an accuracy
NTrees = [100 200 300 400 500];

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
    NTreeStar = NTrees(NLeaveStarId);
end

%% RF-2

% 9.6%
maxDepths = [1024];
Ms = [64];
F1s = [20];

for jj = 1:1:numberOfExperiments
    setSeed(28111993*jj);

    fprintf('%d : Split the data\n', jj);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr, false);
    
    idxCV = splitGetCV(K, length(TTr.y));
    
    % K-fold
    for kk=1:1:K
        fprintf('%d : %dth fold\n', j, kk);
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, kk, false);
        
        for i = 1:1:length(maxDepths)
            for j = 1:1:length(Ms)
                for k = 1:1:length(F1s)
                    md = maxDepths(i);
                    m = Ms(j);
                    f1 = F1s(k);
                    pTrain={'maxDepth',md,'M',m,'F1',f1,'minChild',5};
                    forest=forestTrain(TTTr.Z, TTTr.y,pTrain{:});

                    hsPr0 = forestApply(single(full(TTTe.Z)),forest);
                    err(kk, i, j, k) = balancedErrorRate(hsPr0, TTTe.y);
                    fprintf('%d %d %d \t\t\t\t\t%f\n', md, m, f1, err(kk, i, j, k));
                end
            end
        end
    end
    fprintf('%f ', mean(err));
    mseTe = mean(err_te);

    [errStar, NTreeStarId] = min(mseTe);
    NTreeStar = NTrees(NLeaveStarId);
end

%% Fernst

% 8.69 %
Ss = [11];
Ms = [2048];

for jj = 1:1:numberOfExperiments
    setSeed(28111993*jj);

    fprintf('%d : Split the data\n', jj);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr, false);
    
    idxCV = splitGetCV(K, length(TTr.y));
    
    % K-fold
    for kk=1:1:K
        fprintf('%d : %dth fold\n', j, kk);
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, kk, false);
        
        for i = 1:1:length(Ss)
            for j = 1:1:length(Ms)
                s = Ss(i);
                m = Ms(j);
                fernPrm=struct('S',s,'M',m,'thrr',[-1 1],'bayes',1);
                [ferns,hsPr0]=fernsClfTrain(TTTr.Z, TTTr.y,fernPrm);
                hsPr1 = fernsClfApply(TTTe.Z, ferns );

                err(kk,i,j) = balancedErrorRate(hsPr1, TTTe.y);
                fprintf('%d %d \t\t\t\t\t%f\n', s, m, err(kk, i, j));
            end
        end
    end
    fprintf('%f ', mean(err));
    mseTe = mean(err_te);

    [errStar, NTreeStarId] = min(mseTe);
    NTreeStar = NTrees(NLeaveStarId);
end
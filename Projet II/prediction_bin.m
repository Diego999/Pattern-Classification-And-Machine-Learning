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

train.y(find(train.y == 2)) = 1;
train.y(find(train.y == 3)) = 1;
train.y(find(train.y == 4)) = 2;

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

% Setup 
t = templateSVM('Solver', 'SMO', 'KernelFunction', 'linear', 'IterationLimit', 1e6, 'KernelScale', 1, 'BoxConstraint', 1);      

% K-fold
for k=1:1:K
    fprintf('%d fold\n', k);
    [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);

    tic
    Mdl = fitcecoc(TTTr.Z, TTTr.y, 'Learners', t, 'Coding', 'onevsall', 'Options', statset('UseParallel', 1));
    CMdl = compact(discardSupportVectors(Mdl));
    yhat = predict(CMdl, TTTe.Z);
    toc

    err(k) = balancedErrorRate(TTTe.y, yhat);
    fprintf('%f ', err(k));
end

% 0.089 +- 0.006

u = mean(err); 
o = std(err);

%% TRAINING FOR PREDICTION

% setup
t = templateSVM('Solver', 'SMO', 'KernelFunction', 'linear', 'IterationLimit', 1e6, 'KernelScale', 1, 'BoxConstraint', 1);      

% Train
tic
Mdl = fitcecoc(Tr.Z, Tr.y, 'Learners', t, 'Coding', 'onevsall', 'Options', statset('UseParallel', 1));
CMdl = compact(discardSupportVectors(Mdl));

% Predict
Ytest = predict(CMdl, Te.Z);
toc

Ytest(find(Ytest == 2)) = 0;

% Write the prediction
save('pred_binary', 'Ytest');
%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

ratio = 0.8;
M = 500;

%% Create data
fprintf('Creating Train & Test sets\n');
tic
[Tr, Te] = createTrainingTestingHOG(train.X_hog, train.y, ratio);
toc

%% Prepare data

fprintf('Prepare the data for the training\n');
tic
[Tr, Te] = prepareDataHOG(Tr, Te, M);
toc

%% Train

fprintf('Training\n');

binaryClassification = false;

yTr = Tr.y;
yTe = Te.y;
    
if binaryClassification    
    yTr(find(yTr == 1)) = 1;
    yTr(find(yTr == 2)) = 1;
    yTr(find(yTr == 3)) = 1;
    yTr(find(yTr == 4)) = 2;
    
    yTe(find(yTe == 1)) = 1;
    yTe(find(yTe == 2)) = 1;
    yTe(find(yTe == 3)) = 1;
    yTe(find(yTe == 4)) = 2;
end

    % Binary
    % Default : 24.18% 

    % Multiclass
    % Default : 27.34% 
    
     t = templateSVM('Solver', 'SMO', 'KernelFunction', 'linear', 'IterationLimit', 1e6, 'KernelScale', 1, 'BoxConstraint', 1);   
    
    % Training using Z 
    tic
    Mdl = fitcecoc(Tr.Z, yTr, 'Learners', t, 'Coding', 'onevsall', 'Options', statset('UseParallel', 1));
    toc
    CMdl = compact(discardSupportVectors(Mdl));
    yhat = predict(CMdl, Te.Z);

    errZ = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error  Z: %.2f%%\n', errZ * 100);
    

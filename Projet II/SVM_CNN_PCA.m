%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

ratio = 0.8;
M = 200;

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

%% Train

fprintf('Training\n');

binaryClassification = true;

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

    % Binary : 9.32% (Z)
    
    % Multiclass : 9.05% (Z)
    
    t = templateSVM('Solver', 'SMO', 'KernelFunction', 'linear', 'IterationLimit', 1e6, 'KernelScale', 1, 'BoxConstraint', 1);   
      
    % Training using Z 
    tic
    Mdl = fitcecoc(Tr.Z, yTr, 'Learners', t, 'Coding', 'onevsall', 'Options', statset('UseParallel', 1));
    toc
    CMdl = compact(discardSupportVectors(Mdl));
    yhat = predict(CMdl, Te.Z);
    
    errZ = balancedErrorRate(yTe, yhat);
    fprintf('\nBER Testing error  Z: %.2f%%\n', errZ * 100);
    
%     % Training using nZ 
%     Mdl = fitcecoc(Tr.nZ, yTr, 'Learners', t, 'Coding', 'onevsall', 'Options', statset('UseParallel', 1));
%     CMdl = compact(discardSupportVectors(Mdl));
%     yhat = predict(CMdl, Te.nZ);
%      
%     errnZ = balancedErrorRate(yTe, yhat);
%     fprintf('\nBER Testing error  nZ: %.2f%%\n', errnZ * 100);

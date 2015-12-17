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
[Tr, Te] = createTrainingTestingCNN(train.X_cnn, train.y, ratio);
toc

%% Prepare data

fprintf('Prepare the data for the training\n');
tic
[Tr, Te] = prepareDataCNN(Tr, Te, M);
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
    % Default : 9.19
    % Linear : ~%
    % RBF : ~%
    % Poly^7 : ~%

    % Multiclass
    % Default : 8.85% (nZ), 10.12% (Z)
    % Linear : ~27.11%
    % RBF : ~26.88%
    % Poly^7 : 26.65%
    
    t = templateSVM();%'Solver', 'ISDA', 'KernelFunction', 'rbf', 'BoxConstraint', Inf);   
      
    % Training using Z 
    Mdl = fitcecoc(Tr.Z, yTr, 'Learners', t, 'Options', statset('UseParallel', 1));
    CMdl = compact(discardSupportVectors(Mdl));
    yhat = predict(CMdl, Te.Z);
    
    errZ = balancedErrorRate(yTe, yhat);
    fprintf('\nBER Testing error  X: %.2f%%\n', errZ * 100);
    
    % Training using nZ 
    Mdl = fitcecoc(Tr.nZ, yTr, 'Learners', t, 'Options', statset('UseParallel', 1));
    CMdl = compact(discardSupportVectors(Mdl));
    yhat = predict(CMdl, Te.nZ);
    
    errnZ = balancedErrorRate(yTe, yhat);
    fprintf('\nBER Testing error  X: %.2f%%\n', errnZ * 100);

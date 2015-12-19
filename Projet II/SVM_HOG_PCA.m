%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

ratio = 0.8;
M = 3000;

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

    % Binary
    % Default : 25.5 (X), 28.73 (nX), 26.08% (Z), 29.59 (nZ)
    % Linear : ~29.96%
    % RBF : ~23.81%
    % Poly^7 : ~20.8%

    % Multiclass
    % Default : 29.59 (X), 30.46 (nX), 29.52% (Z) 31.05 (nZ)
    % Linear : ~31.06%
    % RBF : ~27.04%
    % Poly^7 : ~25.6
    
    t = templateSVM();%'Solver', 'ISDA', 'KernelFunction', 'polynomial', 'PolynomialOrder', 7, 'BoxConstraint', Inf);
    
    % Training using X
    Mdl = fitcecoc(Tr.X, yTr, 'Learners', t, 'Options', statset('UseParallel', 1));
    CMdl = compact(discardSupportVectors(Mdl));
    yhat = predict(CMdl, Te.X);

    errX = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error  X: %.2f%%\n', errX * 100);

    % Training using nX
    Mdl = fitcecoc(Tr.nX, yTr, 'Learners', t, 'Options', statset('UseParallel', 1));
    CMdl = compact(discardSupportVectors(Mdl));
    yhat = predict(CMdl, Te.nX);

    errnX = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error nX: %.2f%%\n', errnX * 100);

    % Training using Z
    Mdl = fitcecoc(Tr.Z, yTr, 'Learners', t, 'Options', statset('UseParallel', 1));
    CMdl = compact(discardSupportVectors(Mdl));
    yhat = predict(CMdl, Te.Z);

    errZ = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error  Z: %.2f%%\n', errZ * 100);
    
    % Training using nZ
    Mdl = fitcecoc(Tr.nZ, yTr, 'Learners', t, 'Options', statset('UseParallel', 1));
    CMdl = compact(discardSupportVectors(Mdl));
    yhat = predict(CMdl, Te.nZ);

    errnZ = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error  nZ: %.2f%%\n', errnZ * 100);

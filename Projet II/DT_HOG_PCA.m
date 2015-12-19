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
    % 27.85 (nX), 27.65% (X)
    % AdaBoostM1, 200, Tree : 23.08% (nX)
    % LogitBoost, 200, Tree : 23.89% (nX)
    % GentleBoost, 200, Tree : 23.79% (nX)
    % RUSBoost, 200, Tree : 29.51% (nX)
    % Bag, 200, Tree, : 25.5% (nX)
    
    % Multiclass
    % 27.04% (nX), 28.75% (X), 61.67% /nZ), 64.63% (Z)
    % AdaBoostM2, 200, Tree : 58.46% (nX)
    % RUSBoost, 200, Tree : 49.67% (nX)
    % Bag, 200, Tree, : 28.66% (nX), 500 : 28.35%
    
    % Training using nX
    CMdl = fitensemble(Tr.nX, yTr, 'Bag', 200, 'Tree', 'Type', 'classification');
    yhat = predict(CMdl, Te.nX);

    errnX = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error nX: %.2f%%\n', errnX * 100);
    
    % Training using X
    CMdl = fitensemble(Tr.X, yTr, 'Bag', 200, 'Tree', 'Type', 'classification');
    yhat = predict(CMdl, Te.X);

    errX = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error X: %.2f%%\n', errX * 100);
    
    % Training using nZ
    CMdl = fitensemble(Tr.nZ, yTr, 'Bag', 200, 'Tree', 'Type', 'classification');
    yhat = predict(CMdl, Te.nZ);

    errnZ = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error nZ: %.2f%%\n', errnZ * 100);
    
    % Training using Z
    CMdl = fitensemble(Tr.Z, yTr, 'Bag', 200, 'Tree', 'Type', 'classification');
    yhat = predict(CMdl, Te.Z);

    errZ = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error Z: %.2f%%\n', errZ * 100);

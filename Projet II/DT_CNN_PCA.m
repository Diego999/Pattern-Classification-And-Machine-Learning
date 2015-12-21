% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

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
    % AdaBoostM1, 200, Tree : 11.14% (nZ), 9.80% (Z)
    % LogitBoost, 200, Tree : 10.68% (nZ), 9.64% (Z)
    % GentleBoost, 200, Tree : 10.95% (nZ), 9.98% (Z)
    % RUSBoost, 200, Tree : 14.31% (nZ), 11.74% (Z)
    % Bag, 200, Tree, : 9.84% (nZ), 9.09% (Z)
    
    % Multiclass
    % AdaBoostM2, 200, Tree : 16.96% (nZ), 15.2% (Z)
    % RUSBoost, 200, Tree : 50.67% (nZ), 50.61% (Z)
    % Bag, 200, Tree, : 10.5% (nZ), 9.88% (Z)
    
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

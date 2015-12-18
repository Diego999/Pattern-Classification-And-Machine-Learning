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
    % AdaBoostM1, 200, Tree : 13.57% (nZ), 8.69% (Z)
    % LogitBoost, 200, Tree : 12.80% (nZ), 9.74% (Z), 2000 16.07% (bZ),
    % 10.52% (Z)
    % GentleBoost, 200, Tree : 14.67% (nZ), 10.71% (Z)
    % RUSBoost, 200, Tree : 17.81% (nZ), 12.25% (Z)
    % Bag, 200, Tree, : 11.51% (nZ), 10.13% (Z)
    
    % Multiclass
    % AdaBoostM2, 200, Tree : 29.63% (bZ), 15.34% (Z)
    % RUSBoost, 200, Tree : 50.94% (nZ), 50.60% (Z)
    % Bag, 200, Tree, : 11.55% (nZ), 11.36 (Z)

     CMdl = fitensemble(Tr.Z, yTr, 'Bag', 200, 'Tree', 'Type', 'classification');
        
    % Training using nZ
    yhat = predict(CMdl, Te.nZ);
 
    errnZ = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error nZ: %.2f%%\n', errnZ * 100);
    
    % Training using Z
    yhat = predict(CMdl, Te.Z);

    errZ = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error Z: %.2f%%\n', errZ * 100);

% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

ratio = 0.8;
M = 1000;

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
    % 100 : 44.30% (nZ), 45.15% (Z), 25.51% (nX), 26.74% (X)
    % 200 : 45.73% (nZ), 42.22% (Z), 24.54% (nX), 24.89% (X)
    % 500 : 45.19% (nZ), 43.37% (Z), 25.33% (nX), 25.64% (X)
    
    % Multiclass
    % 100 : 63.13% (nZ), 65.14% (Z), 30.52% (nX), 29.65% (X)
    % 200 : 64.65% (nZ), 64.97% (Z), 28.85% (nX), 28.08% (X)
    % 500 : 68.24% (nZ), 65.72% (Z), 28.24% (nX), 28.39% (X)
    
    % Training using nZ
    BaggedEnsemble = TreeBagger(200, Tr.nZ, yTr);
    yhat = str2double(predict(BaggedEnsemble, Te.nZ));
 
    errnZ = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error nZ: %.2f%%\n', errnZ * 100);
    
    % Training using Z
    BaggedEnsemble = TreeBagger(200, Tr.Z, yTr);
    yhat = str2double(predict(BaggedEnsemble, Te.Z));

    errZ = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error Z: %.2f%%\n', errZ * 100);
    
    % Training using nX
    BaggedEnsemble = TreeBagger(200, Tr.nX, yTr);
    yhat = str2double(predict(BaggedEnsemble, Te.nX));
 
    errnX = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error nX: %.2f%%\n', errnX * 100);
    
    % Training using X
    BaggedEnsemble = TreeBagger(200, Tr.X, yTr);
    yhat = str2double(predict(BaggedEnsemble, Te.X));

    errX = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error X: %.2f%%\n', errX * 100);

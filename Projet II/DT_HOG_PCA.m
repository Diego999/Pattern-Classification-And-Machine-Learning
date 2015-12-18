%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

ratio = 0.8;
M = 3000; % All the data

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
    % AdaBoostM1, 200, Tree : 23.08% (nX)
    % LogitBoost, 200, Tree : 23.89% (nX)
    % GentleBoost, 200, Tree : 23.79% (nX)
    % RUSBoost, 200, Tree : 29.51% (nX)
    % Bag, 200, Tree, : 25.5% (nX)
    
    % Multiclass
    % AdaBoostM2, 200, Tree : 58.46% (nX)
    % RUSBoost, 200, Tree : 49.67% (nX)
    % Bag, 200, Tree, : 28.66% (nX), 500 : 28.35%
    
    %CMdl = fitensemble(Tr.nX, yTr, 'Bag', 200, 'Tree', 'Type', 'classification');
    % Training using nX
    Mdl = fitcecoc(Tr.nX, yTr, 'Learners', t, 'Options', statset('UseParallel', 1));
    CMdl = compact(discardSupportVectors(Mdl));
    yhat = predict(CMdl, Te.nX);

    errnX = balancedErrorRate(yTe, yhat);
    fprintf('BER Testing error nX: %.2f%%\n', errnX * 100);

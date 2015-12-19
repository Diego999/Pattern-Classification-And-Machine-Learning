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
    % 200 : 9.29% (nZ), 9.26% (Z)
    % 500 : 9.6% (nZ), 9.31% (Z)
    % 1000 : 9.73% (nZ), 9.07% (Z)
    
    % Multiclass
    % 200 : % 11.27% (nZ), 11.44% (Z)
    % 500 : 11.64% (nZ), 11.12% (Z)
    % 1000 : 11.08% (nZ), 10.90% (Z)
    
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

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
% 20.61% (nZ), 21.40% (Z)
% Multi
% 21.51% (nZ), 22.35% (Z)

fernPrm = struct('S', 10, 'M', 50, 'thrr', [-1 1], 'bayes', 1);

% Training using Z
ferns = fernsClfTrain(Tr.Z, yTr, fernPrm);
yhat = fernsClfApply(Te.Z, ferns );
  
errZ = balancedErrorRate(yTe, yhat);

% Training using nZ
ferns = fernsClfTrain(Tr.nZ, yTr, fernPrm);
yhat = fernsClfApply(Te.nZ, ferns );
  
errnZ = balancedErrorRate(yTe, yhat);

fprintf('BER Testing error Z: %.2f%%\n', errZ * 100);
fprintf('BER Testing error nZ: %.2f%%\n', errnZ * 100);

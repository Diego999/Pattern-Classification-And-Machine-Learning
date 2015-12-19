%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

ratio = 0.8;
M = 150;

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

yTr = Tr.y;
yTe = Te.y;
      
yTr(find(yTr == 1)) = 1;
yTr(find(yTr == 2)) = 1;
yTr(find(yTr == 3)) = 1;
yTr(find(yTr == 4)) = 2;

yTe(find(yTe == 1)) = 1;
yTe(find(yTe == 2)) = 1;
yTe(find(yTe == 3)) = 1;
yTe(find(yTe == 4)) = 2;
    
% 8.85% (Z), 8.43% (nZ)

pBoost=struct('nWeak',256,'verbose',16,'pTree',struct('maxDepth',4));

% Training using Z
model = adaBoostTrain(Tr.Z(yTr==1, :), Tr.Z(yTr==2,:), pBoost);

yp = yTe(find(yTe == 1));
yn = yTe(find(yTe == 2));

fp = adaBoostApply(Te.Z(yTe==1, :), model);
fp = double(fp > 0);
fp = fp + ones(length(fp), 1);
fn = adaBoostApply(Te.Z(yTe==2, :), model);
fn = double(fn > 0);
fn = fn + ones(length(fn), 1);

yhat = [fp; fn];
ytrue = [yp; yn];
errZ = balancedErrorRate(ytrue, yhat);

% Training using nZ
model = adaBoostTrain(Tr.nZ(yTr==1, :), Tr.nZ(yTr==2,:), pBoost);

yp = yTe(find(yTe == 1));
yn = yTe(find(yTe == 2));

fp = adaBoostApply(Te.nZ(yTe==1, :), model);
fp = double(fp > 0);
fp = fp + ones(length(fp), 1);
fn = adaBoostApply(Te.nZ(yTe==2, :), model);
fn = double(fn > 0);
fn = fn + ones(length(fn), 1);

yhat = [fp; fn];
ytrue = [yp; yn];

errnZ = balancedErrorRate(ytrue, yhat);


fprintf('BER Testing error Z: %.2f%%\n', errZ * 100);
fprintf('BER Testing error nZ: %.2f%%\n', errnZ * 100);

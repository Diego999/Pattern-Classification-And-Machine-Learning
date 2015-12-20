%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

ratio = 0.8;
M = 500;

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
    
% 10.63% (Z), 12.55% (nZ)

pBoost=struct('nWeak',256,'verbose',16,'pTree',struct('maxDepth',4));

yp = yTe(find(yTe == 1));
yn = yTe(find(yTe == 2));

% Training using Z
model = adaBoostTrain(Tr.Z(yTr==1, :), Tr.Z(yTr==2,:), pBoost);

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

fp = adaBoostApply(Te.nZ(yTe==1, :), model);
fp = double(fp > 0);
fp = fp + ones(length(fp), 1);
fn = adaBoostApply(Te.nZ(yTe==2, :), model);
fn = double(fn > 0);
fn = fn + ones(length(fn), 1);

yhat = [fp; fn];
ytrue = [yp; yn];

errnZ = balancedErrorRate(ytrue, yhat);

% Training using X
model = adaBoostTrain(Tr.X(yTr==1, :), Tr.X(yTr==2,:), pBoost);

fp = adaBoostApply(Te.X(yTe==1, :), model);
fp = double(fp > 0);
fp = fp + ones(length(fp), 1);
fn = adaBoostApply(Te.X(yTe==2, :), model);
fn = double(fn > 0);
fn = fn + ones(length(fn), 1);

errX = balancedErrorRate(ytrue, yhat);

% Training using nX
model = adaBoostTrain(Tr.nX(yTr==1, :), Tr.nX(yTr==2,:), pBoost);

fp = adaBoostApply(Te.nX(yTe==1, :), model);
fp = double(fp > 0);
fp = fp + ones(length(fp), 1);
fn = adaBoostApply(Te.nX(yTe==2, :), model);
fn = double(fn > 0);
fn = fn + ones(length(fn), 1);

errnX = balancedErrorRate(ytrue, yhat);

fprintf('BER Testing error Z: %.2f%%\n', errZ * 100);
fprintf('BER Testing error nZ: %.2f%%\n', errnZ * 100);
fprintf('BER Testing error X: %.2f%%\n', errX * 100);
fprintf('BER Testing error nX: %.2f%%\n', errnX * 100);

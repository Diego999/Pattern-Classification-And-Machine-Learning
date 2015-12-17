%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

ratio = 0.8;
M = 5408; % All the data

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

% Setup NN.
inputSize  = M;
innerSize  = 100;
numepochs = 15;
batchsize = 100;
learningRate = 2;
binaryClassification = false;

% Train using Z
[errZ, nnPredZ] = neuralNetworks(Tr.nZ, Tr.y, Te.nZ, Te.y, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

% Train using Z
inputSize = size(Tr.nZU, 2);
[errZU, nnPredZU] = neuralNetworks(Tr.nZU, Tr.y, Te.nZU, Te.y, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

% Train using X
inputSize = size(Tr.nX, 2);
[errX, nnPredX] = neuralNetworks(Tr.nX, Tr.y, Te.nX, Te.y, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

fprintf('\nBER Testing error  Z: %.2f%%\n', errZ * 100);
fprintf('\nBER Testing error  ZU: %.2f%%\n', errZU * 100);
fprintf('\nBER Testing error  X: %.2f%%\n', errX * 100);

% figure('Name', ['NN on HOG + PCA, M = ' num2str(M)]);
% subplot(131);
% imagesc(nnPredZ); colorbar;
% title(['BER(Z) = ' num2str(errZ)]);
% subplot(132);
% imagesc(nnPredZU); colorbar;
% title(['BER(ZU) = ' num2str(errZU)]);
% subplot(133);
% imagesc(nnPredX); colorbar;
% title(['BER(X) = ' num2str(errX)]);

% NOTE: each subplot represent the same information but computed from a
%       different input: each row is a sample of the test set, each
%       column corresponds to a classification label and the color
%       represents the probability of being in this or that category.

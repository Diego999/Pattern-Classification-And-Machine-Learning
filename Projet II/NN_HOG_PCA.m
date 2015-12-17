%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

ratio = 0.8;
M = 1000; % All the data

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

yTr = Tr.y;
yTe = Te.y;
    
if binaryClassification    
    yTr(find(yTr == 2)) = 1;
    yTr(find(yTr == 3)) = 1;
    yTr(find(yTr == 4)) = 2;
    
    yTe(find(yTe == 2)) = 1;
    yTe(find(yTe == 3)) = 1;
    yTe(find(yTe == 4)) = 2;
end

% Train using Z
[errZ, nnPredZ] = neuralNetworks(Tr.nZ, yTr, Te.nZ, yTe, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

% Train using X
inputSize = size(Tr.nX, 2);
[errX, nnPredX] = neuralNetworks(Tr.nX, yTr, Te.nX, yTe, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

if binaryClassification
   fprintf('\n Binary classification\n');
else
   fprintf('\n Multiclass classification\n');
end

fprintf('\nBER Testing error  Z: %.2f%%\n', errZ * 100);
fprintf('\nBER Testing error  X: %.2f%%\n', errX * 100);

% figure('Name', ['NN on HOG + PCA, M = ' num2str(M)]);
% subplot(131);
% imagesc(nnPredZ); colorbar;
% title(['BER(Z) = ' num2str(errZ)]);
% subplot(132);
% imagesc(nnPredX); colorbar;
% title(['BER(X) = ' num2str(errX)]);

% NOTE: each subplot represent the same information but computed from a
%       different input: each row is a sample of the test set, each
%       column corresponds to a classification label and the color
%       represents the probability of being in this or that category.

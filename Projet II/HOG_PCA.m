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
outputSize = 4;

nn = nnsetup([inputSize innerSize outputSize]);

opts.numepochs  = 15;
opts.batchsize  = 100;
opts.plot       = 1;
nn.learningRate = 2;

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = opts.batchsize * floor(size(Tr.nZ) / opts.batchsize);
Tr.nZ     = Tr.nZ(1:numSampToUse, :);
labels       = Tr.y(1:numSampToUse);

% prepare labels for NN
LL = [1 * (labels == 1), ... % first column, p(y=1)
      1 * (labels == 2), ... % second column, p(y=2), etc
      1 * (labels == 3), ...
      1 * (labels == 4) ];
[nn, ~] = nntrain(nn, Tr.nZ, LL, opts);

% to get the scores we need to do nnff (feed-forward)
nn.testing = 1;
nn = nnff(nn, Te.nZ, zeros(size(Te.nZ, 1), nn.size(end)));
nn.testing = 0;

% predict on the test set
nnPredZ = nn.a{end};

% get the most likely class
[~, predictionsZ] = max(nnPredZ, [], 2);

% Apply the same NN with different inputs: ZU
inputSize = size(Tr.nZU, 2);
nn = nnsetup([inputSize innerSize outputSize]);
nn.learningRate = 2;
[nn, ~] = nntrain(nn, Tr.nZU, LL, opts);
nn.testing = 1;
nn = nnff(nn, Te.nZU, zeros(size(Te.nZU, 1), nn.size(end)));
nn.testing = 0;
nnPredZU = nn.a{end};
% get the most likely class
[~, predictionsZU] = max(nnPredZU, [], 2);

% Apply the same NN with different inputs: X
inputSize = size(Tr.nX, 2);
nn = nnsetup([inputSize innerSize outputSize]);
nn.learningRate = 2;
[nn, ~] = nntrain(nn, Tr.nX, LL, opts);
nn.testing = 1;
nn = nnff(nn, Te.nX, zeros(size(Te.nX, 1), nn.size(end)));
nn.testing = 0;
nnPredX = nn.a{end};
% get the most likely class
[~, predictionsX] = max(nnPredX, [], 2);

berErrZ = balancedErrorRate(Te.y, predictionsZ);
berErrZU = balancedErrorRate(Te.y, predictionsZU);
berErrX = balancedErrorRate(Te.y, predictionsX);
fprintf('\nBER Testing error  Z: %.2f%%\n', berErrZ * 100);
fprintf('\nBER Testing error ZU: %.2f%%\n', berErrZU * 100);
fprintf('\nBER Testing error  X: %.2f%%\n', berErrX * 100);

figure('Name', ['NN on HOG + PCA, M = ' num2str(M)]);
subplot(131);
imagesc(nnPredZ); colorbar;
title(['BER(Z) = ' num2str(berErrZ)]);
subplot(132);
imagesc(nnPredZU); colorbar;
title(['BER(ZU) = ' num2str(berErrZU)]);
subplot(133);
imagesc(nnPredX); colorbar;
title(['BER(X) = ' num2str(berErrX)]);

% NOTE: each subplot represent the same information but computed from a
%       different input: each row is a sample of the test set, each
%       column corresponds to a classification label and the color
%       represents the probability of being in this or that category.

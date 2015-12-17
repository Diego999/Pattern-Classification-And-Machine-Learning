%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

ratio = 0.8;
M = 100;

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

% Setup NN.
inputSize  = M;
innerSize  = 50;
outputSize = 4;

nn = nnsetup([inputSize innerSize outputSize]);

opts.numepochs  = 15;
opts.batchsize  = 100;
opts.plot       = 1;
nn.learningRate = 2;

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = opts.batchsize * floor(size(Tr.nZ) / opts.batchsize);
Tr.nZ = Tr.nZ(1:numSampToUse, :);
labels = Tr.y(1:numSampToUse);

% prepare labels for NN
LL = [1 * (labels == 1), ... % first column, p(y=1)
      1 * (labels == 2), ... % second column, p(y=2), etc
      1 * (labels == 3), ...
      1 * (labels == 4) ];

tstart = tic; % because nntrain uses tic too...
[nn, ~] = nntrain(nn, Tr.nZ, LL, opts);
toc(tstart)

% to get the scores we need to do nnff (feed-forward)
tstart = tic;
nn.testing = 1;
nn = nnff(nn, Te.nZ, zeros(size(Te.nZ, 1), nn.size(end)));
nn.testing = 0;

% predict on the test set
nnPredZ = nn.a{end};

% get the most likely class
[~, predictionsZ] = max(nnPredZ, [], 2);
toc(tstart)

% Apply the same NN with different inputs: ZU
% NOTE: not doing this as the matrixes are HUGE!!!

% Apply the same NN with different inputs: X
fprintf('Applaying NN on original data...\n');
tstart = tic;
inputSize = size(Tr.X, 2); % NOT Tr.normX as it is HUGE and NOT sparse
nn = nnsetup([inputSize innerSize outputSize]);
nn.learningRate = 2;
[nn, ~] = nntrain(nn, Tr.X, LL, opts);
nn.testing = 1;
nn = nnff(nn, Te.X, zeros(size(Te.X, 1), nn.size(end)));
nn.testing = 0;
nnPredX = nn.a{end};
% get the most likely class
[~, predictionsX] = max(nnPredX, [], 2);
toc(tstart)

berErrZ = balancedErrorRate(Te.y, predictionsZ);
berErrX = balancedErrorRate(Te.y, predictionsX);
fprintf('\nBER Testing error  Z: %.2f%%\n', berErrZ * 100);
fprintf('\nBER Testing error  X: %.2f%%\n', berErrX * 100);

figure('Name', ['NN on CNN + PCA, M = ' num2str(M)]);
subplot(131);
imagesc(nnPredZ); colorbar;
title(['BER(Z) = ' num2str(berErrZ)]);
subplot(132);
imagesc(nnPredX); colorbar;
title(['BER(X) = ' num2str(berErrX)]);
subplot(133);
imagesc(abs(nnPredX - nnPredZ)); colorbar;
title('diff');

% NOTE: each subplot represent the same information but computed from a
%       different input: each row is a sample of the test set, each
%       column corresponds to a classification label and the color
%       represents the probability of being in this or that category.

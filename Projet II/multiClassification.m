%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

numberOfExperiments = 10;
proportionOfTraining = 0.8;
K = 10;
M = 150;

%% Create data
fprintf('Creating Train & Test sets\n');
tic
[Tr, Te] = createTrainingTestingCNN(train.X_cnn, train.y, proportionOfTraining);
toc

%% Prepare data

fprintf('Prepare the data for the training\n');
tic
[Tr, Te] = prepareDataCNN(Tr, Te, M);
toc

%%
% 75%
%**********************************
%            TEST 1
%**********************************

fprintf('Test 1\n')
for i = 1:numberOfExperiments
    fprintf('%d ', i);
    setSeed(28111993*i);
    out = 1;
    err1(i) = balancedErrorRate(Te.y, repmat([out], length(Tr.y), 1));
end
fprintf('\n%f\n', mean(err1));
saveFile(err1, 'results/multi/err1');

%% 9.14 %
%**********************************
%            TEST 2
%**********************************

N = length(Tr.y);

% Setup NN.
inputSize  = M;
innerSize  = 700;
numepochs = 20;
batchsize = 100;
learningRate = 2;
binaryClassification = false;

Tr_ = Tr;
Te_ = Te;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
        
    idxCV = splitGetCV(K, length(TTr.y));
    
    % K-fold
    for k=1:1:K
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);
        
        [errnZ, nnPrednZ] = neuralNetworks(TTTr.nZ, TTTr.y, TTTe.nZ, TTTe.y, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

        err_te(k) = errnZ;
    end
    err2(j) = mean(err_te);
end
fprintf('\n%f\n', mean(err2));
saveFile(err2, 'results/multi/err2');

%% 10.39%
%**********************************
%            TEST 3
%**********************************

N = length(Tr.y);

% Setup 
NLeaves = 500;
depth = 512;
Tr_ = Tr;
Te_ = Te;

t = templateTree('MaxNumSplits', depth);

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
        
    idxCV = splitGetCV(K, length(TTr.y));
    tic
    % K-fold
    for k=1:1:K
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);
        
        CMdl = fitensemble(TTTr.Z, TTTr.y, 'Bag', NLeaves, 'Tree', 'Type', 'classification', 'Learners', t);
        yhat = predict(CMdl, TTTe.Z);

        err_te(k) = balancedErrorRate(TTTe.y, yhat);
    end
    toc
    err3(j) = mean(err_te);
end
fprintf('\n%f\n', mean(err3));
saveFile(err3, 'results/multi/err3');

%% 10.41%
%**********************************
%            TEST 4
%**********************************

N = length(Tr.y);

% Setup 
NTrees = 500;

Tr_ = Tr;
Te_ = Te;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
        
    idxCV = splitGetCV(K, length(TTr.y));
    tic
    % K-fold
    for k=1:1:K
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);
        
        BaggedEnsemble = TreeBagger(NTrees, TTTr.Z, TTTr.y);
        yhat = str2double(predict(BaggedEnsemble, TTTe.Z));

        err_te(k) = balancedErrorRate(TTTe.y, yhat);
    end
    toc
    err4(j) = mean(err_te);
end
fprintf('\n%f\n', mean(err4));
saveFile(err4, 'results/multi/err4');

%% 8.71%
%**********************************
%            TEST 5
%**********************************

% Setup 
t = templateSVM('Solver', 'SMO', 'KernelFunction', 'linear', 'IterationLimit', 1e6, 'KernelScale', 1, 'BoxConstraint', 1);      

Tr_ = Tr;
Te_ = Te;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr_, false);
        
    idxCV = splitGetCV(K, length(TTr.y));
      
    % K-fold
    for k=1:1:K
        [TTTr, TTTe] = splitGetTrTe(TTr, idxCV, k, false);
        Mdl = fitcecoc(TTTr.Z, TTTr.y, 'Learners', t, 'Coding', 'onevsall', 'Options', statset('UseParallel', 1));
        CMdl = compact(discardSupportVectors(Mdl));
        yhat = predict(CMdl, TTTe.Z);
    
        err_te(k) = balancedErrorRate(TTTe.y, yhat);
    end
    
    err5(j) = err_te;
end

fprintf('\n%f\n', mean(err5));
saveFile(err5, 'results/multi/err5');

%% Plot
s = [1 numberOfExperiments];

err1 = openFile('results/multi/err1', s);
err2 = openFile('results/multi/err2',s);
err3 = openFile('results/multi/err3', s);
err4 = openFile('results/multi/err4', s);
err5 = openFile('results/multi/err5', s);

figure;
boxplot([err1' err2' err3' err4' err5']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 ', ...
'2 ', ...
'3 ', ...
'4 ', ...
'5 ');
set(gca, 'XGrid','on')
set(gca, 'YGrid','on')
set(gca,'LineWidth',1.5);
ylim([0 0.8])
set(gca,'YTick',0:0.05:0.8)
xlabel('Model');
ylabel('BER');
%print('../report/figures/models','-djpeg','-noui')

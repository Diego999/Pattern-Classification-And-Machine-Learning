%% Initialization
clear all;
close all;

addpath(genpath('Piotr'));
addpath(genpath('DeepLearnToolbox'));

load train/train.mat;

numberOfExperiments = 30;
proportionOfTraining = 0.8;
K = 10;
M = 150;

%% Create data
fprintf('Creating Train & Test sets\n');
tic
[Tr, Te] = createTrainingTestingCNN(train.X_cnn, train.y, 1.0);
toc

%% Prepare data

fprintf('Prepare the data for the training\n');
tic
[Tr, Te] = prepareDataCNN(Tr, Te, M);
toc

%% Constant 75%
%**********************************
%            TEST 1
%**********************************

fprintf('Test 1\n')
for i = 1:numberOfExperiments
    fprintf('%d ', i);
    setSeed(28111993*i);
    out = 1;
    err1(i) = 0.75;%balancedErrorRate(Te.y, repmat([out], length(Tr.y), 1));
end
fprintf('\n%f +- 0\n', mean(err1));
saveFile(err1, 'results/multi/err1');

%% Neural Network 8.646 +- 0.008 %
%**********************************
%            TEST 2
%**********************************

% Setup NN.
inputSize  = M;
innerSize  = 700;
numepochs = 20;
batchsize = 100;
learningRate = 2;
binaryClassification = false;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr, false);
        
    [errnZ, nnPrednZ] = neuralNetworks(TTr.nZ, TTr.y, TTe.nZ, TTe.y, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

    err2(j) = errnZ;
end
fprintf('\n%f +- %f\n', mean(err2), std(err2));
saveFile(err2, 'results/multi/err2');

%% bootstrap aggregation 10.174 +- 0.006
%**********************************
%            TEST 3
%**********************************

% Setup 
NLeaves = 500;
depth = 512;

t = templateTree('MaxNumSplits', depth);

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr, false);
    
    tic
    CMdl = fitensemble(TTr.Z, TTr.y, 'Bag', NLeaves, 'Tree', 'Type', 'classification', 'Learners', t);
    yhat = predict(CMdl, TTe.Z);
    toc
    
    err3(j) = balancedErrorRate(TTe.y, yhat);
end

fprintf('\n%f +- %f\n', mean(err3), std(err3));
saveFile(err3, 'results/multi/err3');

%% bootstrap aggregation2 10.210 +- 0.006
%**********************************
%            TEST 4
%**********************************

% Setup 
NTrees = 500;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr, false);

    tic
    BaggedEnsemble = TreeBagger(NTrees, TTr.Z, TTr.y);
    yhat = str2double(predict(BaggedEnsemble, TTe.Z));
    toc
    
    err4(j) = balancedErrorRate(TTe.y, yhat);
end
fprintf('\n%f +- %f\n', mean(err4), std(err4));
saveFile(err4, 'results/multi/err4');

%% SVM 8.71 TODO % 75'
%**********************************
%            TEST 5
%**********************************

% Setup 
t = templateSVM('Solver', 'SMO', 'KernelFunction', 'linear', 'IterationLimit', 1e6, 'KernelScale', 1, 'BoxConstraint', 1);      

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr, false);

    tic
    Mdl = fitcecoc(TTr.Z, TTr.y, 'Learners', t, 'Coding', 'onevsall', 'Options', statset('UseParallel', 1));
    CMdl = compact(discardSupportVectors(Mdl));
    yhat = predict(CMdl, TTe.Z);
    toc
    
    err5(j) = balancedErrorRate(TTe.y, yhat);
end

fprintf('\n%f +- %f\n', mean(err5), std(err5));
saveFile(err5, 'results/multi/err5');

%% Random forest 12.022 +- 0.010
%**********************************
%            TEST 6
%**********************************

% Setup 
maxDepth = 1024;
m = 64;
F1 = 20;

pTrain={'maxDepth',maxDepth,'M',m,'F1',F1,'minChild',5};

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr, false);

    tic
    forest=forestTrain(TTr.Z, TTr.y, pTrain{:});
    yhat = forestApply(single(full(TTe.Z)), forest);
    toc
    
    err6(j) = balancedErrorRate(TTe.y, yhat);
end

fprintf('\n%f +- %f\n', mean(err6), std(err6));
saveFile(err6, 'results/multi/err6');

%% Fernst 11.818 +- 0.009%
%**********************************
%            TEST 7
%**********************************

% Setup 
s = 11;
m = 2048;
fernPrm=struct('S',s,'M',m,'thrr',[-1 1],'bayes',1);

pTrain={'maxDepth',maxDepth,'M',m,'F1',F1,'minChild',5};

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr, false);

    tic    
    [ferns]=fernsClfTrain(TTr.Z, TTr.y, fernPrm);
    yhat = fernsClfApply(TTe.Z, ferns);        
    toc
    
    err7(j) = balancedErrorRate(TTe.y, yhat);
end

fprintf('\n%f +- %f\n', mean(err7), std(err7));
saveFile(err7, 'results/multi/err7');

%% Plot
s = [1 numberOfExperiments];

err1 = openFile('results/multi/err1', s);
err2 = openFile('results/multi/err2',s);
err3 = openFile('results/multi/err3', s);
err4 = openFile('results/multi/err4', s);
err5 = openFile('results/multi/err5', s);
err6 = openFile('results/multi/err6', s);
err7 = openFile('results/multi/err7', s);

figure;
boxplot([err1' err2' err3' err4' err5' err6' err7']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 ', ...
'2 ', ...
'3 ', ...
'4 ', ...
'5 ', ...
'6 ', ...
'7 ');
set(gca, 'XGrid','on')
set(gca, 'YGrid','on')
set(gca,'LineWidth',1.5);
ylim([0 0.8])
set(gca,'YTick',0:0.05:0.8)
xlabel('Model');
ylabel('BER');
%print('../report/figures/models','-djpeg','-noui')

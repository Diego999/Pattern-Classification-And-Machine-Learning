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

train.y(find(train.y == 2)) = 1;
train.y(find(train.y == 3)) = 1;
train.y(find(train.y == 4)) = 2;

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

%% Constant 50%
%**********************************
%            TEST 1
%**********************************

fprintf('Test 1\n')
for i = 1:numberOfExperiments
    fprintf('%d ', i);
    setSeed(28111993*i);
    out = 1;
    err1(i) = 0.5;%balancedErrorRate(Te.y, repmat([out], length(Tr.y), 1));
end
fprintf('\n%f\n', mean(err1));
saveFile(err1, 'results/binary/err1');

%% Neural Network 0.0905 +- 0.009%
%**********************************
%            TEST 2
%**********************************

N = length(Tr.y);

% Setup NN.
inputSize  = M;
innerSize  = 1000;
numepochs = 20;
batchsize = 100;
learningRate = 3;
binaryClassification = true;

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr, false);
        
    [errnZ, nnPrednZ] = neuralNetworks(TTr.nZ, TTr.y, TTe.nZ, TTe.y, inputSize, innerSize, numepochs, batchsize, learningRate, binaryClassification);

    err2(j) = errnZ;
end
fprintf('\n%f +- %f\n', mean(err2), std(err2));
saveFile(err2, 'results/binary/err2');

%% bootstrap aggregation 0.0942 +- 0.007%
%**********************************
%            TEST 3
%**********************************

% Setup 
NLeaves = 200;
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
saveFile(err3, 'results/binary/err3');

%% bootstrap aggregation2  0.925 +- 0.008 %
%**********************************
%            TEST 4
%**********************************

% Setup 
NTrees = 400;

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
saveFile(err4, 'results/binary/err4');

%% AdaBoost Tree 0.799 +- 0.008%
%**********************************
%            TEST 5
%**********************************

% Setup 
nbWeak = 1024;
maxDepth = 4;
pBoost=struct('nWeak',nbWeak,'pTree',struct('maxDepth',maxDepth));

for j = 1:1:numberOfExperiments
    setSeed(28111993*j);

    fprintf('%d ', j);
    [TTr, TTe] = splitProp(proportionOfTraining, Tr, false);
    
    model = adaBoostTrain(TTr.nZ(TTr.y==1,:), TTr.nZ(TTr.y==2,:), pBoost);

    yp = TTe.y(find(TTe.y == 1));
    yn = TTe.y(find(TTe.y == 2));

    fp = adaBoostApply(TTe.nZ(TTe.y==1, :), model);
    fp = double(fp > 0);
    fp = fp + ones(length(fp), 1);
    
    fn = adaBoostApply(TTe.nZ(TTe.y==2, :), model);
    fn = double(fn > 0);
    fn = fn + ones(length(fn), 1);

    yhat = [fp; fn];
    ytrue = [yp; yn];

    err5(j) = balancedErrorRate(ytrue, yhat);
end

fprintf('\n%f +- %f\n', mean(err5), std(err5));
saveFile(err5, 'results/binary/err5');

%% Random forest 0.122 +- 0.010%
%**********************************
%            TEST 6
%**********************************

% Setup 
maxDepth = 2*512;
m = 256;
F1 = 50;

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
saveFile(err6, 'results/binary/err6');

%% Fernst 0.171 +- 0.13%
%**********************************
%            TEST 7
%**********************************

% Setup 
s = 15;
m = 4096;
fernPrm=struct('S',s,'M',m,'thrr',[-1 1],'bayes',1);

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
saveFile(err7, 'results/binary/err7');

%% SVM 0.0886 +- 0.006
%**********************************
%            TEST 8
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
    
    err8(j) = balancedErrorRate(TTe.y, yhat);
end

fprintf('\n%f +- %f\n', mean(err8), std(err8));
saveFile(err8, 'results/binary/err8');

%% Plot
s = [1 numberOfExperiments];

err1 = openFile('results/binary/err1', s);
err2 = openFile('results/binary/err2',s);
err3 = openFile('results/binary/err3', s);
err4 = openFile('results/binary/err4', s);
err5 = openFile('results/binary/err5', s);
err6 = openFile('results/binary/err6', s);
%err7 = openFile('results/binary/err7', s);
err8 = openFile('results/binary/err8', s);


figure;
boxplot([err1' err2' err3' err4' err5' err6' err8']);
h_legend = legend(findobj(gca,'Tag','Box'), ...
'1 Baseline', ...
'2 Neural networks', ...
'3 Bagging1', ...
'4 Bagging2', ...
'5 AdaBoosting', ...
'6 Random forests', ...
'7 SVM');
set(gca, 'XGrid','on')
set(gca, 'YGrid','on')
set(gca,'LineWidth',1.5);
ylim([0 0.8])
set(gca,'fontsize', 18);
set(gca,'YTick',0:0.05:0.8)
xlabel('Model');
ylabel('BER');
print('./report/figures/binaryclassifications.jpg','-djpeg','-noui')
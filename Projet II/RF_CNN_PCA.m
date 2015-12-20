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

binaryClassification = false;

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
%%
    % Binary
    % 200 : 9.29% (nZ), 9.26% (Z)
    % 500 : 9.6% (nZ), 9.31% (Z)
    % 1000 : 9.73% (nZ), 9.07% (Z)

    % Multiclass
    % 200 : % 11.27% (nZ), 11.44% (Z)
    % 500 : 11.64% (nZ), 11.12% (Z)
    % 1000 : 11.08% (nZ), 10.90% (Z)

    maxDepths = [64 128 256 512 1024 2048 4096];
    Ms = [256 512 768 1024];
    F1s = [25 50 100 150];
    
    for i = 1:1:length(maxDepths)
        for j = 1:1:length(Ms)
            for k = 1:1:length(F1s)
                md = maxDepths(i);
                m = Ms(j);
                f1 = F1s(k);
                pTrain={'maxDepth',md,'M',m,'F1',f1,'minChild',5};
                tic, forest=forestTrain(Tr.Z,yTr,pTrain{:}); toc

                hsPr0 = forestApply(single(full(Te.Z)),forest);
                err(i,j,k) = balancedErrorRate(hsPr0, yTe);
               fprintf('%d %d %d \t\t\t\t\t%f\n', md, m, f1, err(i,j,k));
            end
        end
    end
%%

    fernPrm=struct('S',15,'M', 4096, 'thrr',[-1 1],'bayes',1);
    tic, [ferns,hsPr0]=fernsClfTrain(Tr.Z, yTr,fernPrm); toc
    tic, hsPr1 = fernsClfApply(Te.Z, ferns ); toc
    balancedErrorRate(hsPr1, yTe)

%%     
%     % Training using nZ
%     BaggedEnsemble = TreeBagger(200, Tr.nZ, yTr);
%     yhat = str2double(predict(BaggedEnsemble, Te.nZ));
% 
%     errnZ = balancedErrorRate(yTe, yhat);
%     fprintf('BER Testing error nZ: %.2f%%\n', errnZ * 100);
% 
%     % Training using Z
%     BaggedEnsemble = TreeBagger(200, Tr.Z, yTr);
%     yhat = str2double(predict(BaggedEnsemble, Te.Z));
% 
%     errZ = balancedErrorRate(yTe, yhat);
%     fprintf('BER Testing error Z: %.2f%%\n', errZ * 100);

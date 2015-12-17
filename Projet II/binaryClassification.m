% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

clear all;
close all;

load train/train.mat;

y = train.y;
X_cnn = train.X_cnn;
X_hog = train.X_hog;

y(find(y == 1)) = 0;
y(find(y == 2)) = 0;
y(find(y == 3)) = 0;
y(find(y == 4)) = 1;

numberOfExperiments = 30;
proportionOfTraining = 0.8;
K = 20;

%%
%**********************************
%            TEST 1
%**********************************

X = X_hog;
fprintf('Test 1\n')
for i = 1:numberOfExperiments
    fprintf('%d ', i);
    setSeed(28111993*i);
    [XTr, yTr, XTe, yTe] = splitProp(proportionOfTraining, y, X);
     
    out = 0;
    
    err1(i) = balancedErrorRate(yTe, repmat([out], length(yTr), 1));

end
fprintf('\n%f\n', mean(err1));
saveFile(err1, 'results/binary/err1');
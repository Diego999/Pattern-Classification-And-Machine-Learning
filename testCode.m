% Run this to make sure your implementation is correct
% An example of writing csv file is given in the end of this file
clear all;

% generating dummy dataset. regression
beta = [1:1:50]';
N = 1000;
D = length(beta);
tX = ones(N,D);
for i = 2:D
   tX(:,i) = randn(1,N) * i;
end

tXNorm=tX-mean(tX(:));
tXNorm=tXNorm/std(tXNorm(:));

yNorm = tXNorm * beta;
y = tX * beta;

% max error
eps = 1e-2;

% test least squares via gradient descent
disp('least squares via gradient descent...');
alpha = 0.01;
tic
tBeta = leastSquaresGD(yNorm, tXNorm, alpha);
toc
assert(all(abs(tBeta - beta) < eps));
disp('OK!');

% testing least squares
disp('least squares...');
tic
tBeta = leastSquares(y, tX);
toc
assert(all(abs(tBeta - beta) < eps));
disp('OK!');
% 
% testing ridge regression
disp('ridge regression...');
lambda = 1e-5;
tic
tBeta = ridgeRegression(y, tX, lambda);
toc
assert(all(abs(tBeta - beta) < eps));
disp('OK!');
% 
% % generate binary data
% y = (y>0);
% 
% % testing logistic regression
% disp('logistic regression...');
% alpha = 1e-3;
% tBeta = logisticRegression(y,tX,alpha);
% tY = 1.0 ./ (1.0 + exp(-tX * tBeta)) > 0.5;
% assert(sum(tY ~= y) / size(y,1) < 0.2);
% disp('OK!');
% 
% % testing penalize logistic regression
% disp('penalized logistic regression...');
% alpha = 1e-3;
% lambda = 1e-2;
% tBeta = penLogisticRegression(y,tX,alpha,lambda);
% tY = 1.0 ./ (1.0 + exp(-tX * tBeta)) > 0.5;
% assert(sum(tY ~= y) / size(y,1) < 0.2);
% disp('OK!');
% 
% % code for writing csv files
% tY = 1.0 ./ (1.0 + exp(-tX * tBeta));
% csvwrite('predictions_classification.csv', tY);
% 
% disp('test finished successfully');

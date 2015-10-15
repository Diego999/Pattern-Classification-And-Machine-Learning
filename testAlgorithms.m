% Run this to make sure your implementation is correct
% An example of writing csv file is given in the end of this file
clear all;

% generating dummy dataset. regression
beta = [1:1:80]';
N = 2800;
D = length(beta)-1;
tX = ones(N,D);
for i = 1:D
   tX(:,i) = randn(1,N) * i;
end

tXNorm = normalizedData(tX);

tX = [ones(N,1) tX];
tXNorm = [ones(N,1) tXNorm];

yNorm = tXNorm * beta;
y = tX * beta;

% max error
eps = 1e-2;

% test least squares via gradient descent
disp('least squares via gradient descent...');
alpha = 0.33;
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

% generate binary data
y = (y>0);
yNorm = (yNorm>0);

% testing logistic regression
disp('logistic regression...');
alpha = 1e-5;
tBeta = logisticRegression(yNorm,tXNorm,alpha);
tic
tY = 1.0 ./ (1.0 + exp(-tXNorm * tBeta)) > 0.5;
toc
assert(sum(tY ~= yNorm) / size(y,1) < 0.2);
disp('OK!');

% % testing penalize logistic regression
disp('penalized logistic regression...');
alpha = 1e-5;
lambda = 0.5;
tic
tBeta2 = penLogisticRegression(yNorm,tXNorm,alpha,lambda);
toc
tY = 1.0 ./ (1.0 + exp(-tXNorm * tBeta2)) > 0.5;
assert(sum(tY ~= yNorm) / size(y,1) < 0.2);
disp('OK!');

% % code for writing csv files
% tY = 1.0 ./ (1.0 + exp(-tX * tBeta));
% csvwrite('predictions_classification.csv', tY);
% 
% disp('test finished successfully');

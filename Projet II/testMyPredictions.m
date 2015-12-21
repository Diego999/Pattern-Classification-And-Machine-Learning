% --- OBJECT DETECTION PREDICTION CHECK ----
%
% Store your predictions in two mat files named 'pred_binary.mat'
% and 'pred_multiclass.mat'.
% This mat file should contain a vector 'Ytest'
% which contains the prediction _score_ for each test sample.
% The size of Ytest must be 11453x1.
% One way to create these matrices is shown below:
%
%   -- assign your predicted scores to Ytest first, then: --
%   save('pred_binary', 'Ytest');
%
% Once you have the file, running this file will check if the sizes are correct or not.


%% == Test all vs other
clear all;
fprintf('==> Checking all vs other\n\n')

load pred_binary.mat;

% exists?
if ~exist('Ytest', 'var')
  error('No Ytest vector found');
end

% check size
[nU,nA] = size(Ytest);
if (nU ~= 11453) && (nA ~= 1)
  error('size of Ytest is incorrect');
end

% make sure it contains scores and not labels
nUniqueVals = length(unique(Ytest(:)));
uniqueVals = sort(unique(Ytest(:)), 'ascend');
if nUniqueVals ~= 2 || any( uniqueVals(:) ~= [0,1]' )
    error('Ytest should be either 0 or 1');
end


fprintf('==> Checking multiclass\n\n')
clearvars;

%% == Test all vs other
load pred_multiclass.mat;

% exists?
if ~exist('Ytest', 'var')
  error('No Ytest vector found');
end

% check size
[nU,nA] = size(Ytest);
if (nU ~= 11453) && (nA ~= 1)
  error('size of Ytest is incorrect');
end

% make sure it contains scores and not labels
nUniqueVals = length(unique(Ytest(:)));
uniqueVals = sort(unique(Ytest(:)), 'ascend');
if nUniqueVals ~= 4 || any( uniqueVals(:) ~= [1,2,3,4]' )
    error('Ytest should be either 1,2,3 or 4');
end

fprintf('\nSuccessful, you can submit!\n\n');

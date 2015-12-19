% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [TTr, TTe] = splitProp(prop, Tr, isHog)
    % split the data into train and test given a proportion
    N = length(Tr.y);

    shuffle = randperm(N);
    
    if isHog
        X = Tr.X(shuffle, :);
        nX = Tr.nX(shuffle, :);
    end
    Z = Tr.Z(shuffle, :);
    nZ = Tr.nZ(shuffle, :);
    y = Tr.y(shuffle, :);
    
	% generate random indices
	idx = randperm(N);
    Ntr = floor(prop * N);
    
	% select few as training and others as testing
	idxTr = idx(1:Ntr);
	idxTe = idx(Ntr+1:end);
	
    % create train-test split
    if isHog
        TTr.X = X(idxTr, :);
        TTr.nX = nX(idxTr, :);
    end
    TTr.Z = Z(idxTr, :);
    TTr.nZ = nZ(idxTr, :);
    TTr.y = y(idxTr, :);
    
    if isHog
        TTe.X = X(idxTe, :);
        TTe.nX = nX(idxTe, :);
    end
    TTe.Z = Z(idxTe, :);
    TTe.nZ = nZ(idxTe, :);
    TTe.y = y(idxTe, :);
end


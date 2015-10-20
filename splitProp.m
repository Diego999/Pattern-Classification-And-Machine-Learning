% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [XTr, yTr, XTe, yTe] = splitProp(K, y, X)
    % split the data into train and test given a proportion
    N = length(y);

    shuffle = randperm(N);
    Xrandom = X(shuffle,:);
    yrandom = y(shuffle,:);
    
	% generate random indices
	idx = randperm(N);
    Ntr = floor(K * N);
	% select few as training and others as testing
	idxTr = idx(1:Ntr);
	idxTe = idx(Ntr+1:end);
	% create train-test split
    XTr = Xrandom(idxTr,:);
    yTr = yrandom(idxTr);
    XTe = Xrandom(idxTe,:);
    yTe = yrandom(idxTe);
end


% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [idxCV] = splitGetCV(K, N)
    % split the data into train and test given a proportion
	setSeed(28111993);
    
    % split data in K fold (we will only create indices)
    setSeed(1);
    idx = randperm(N);
    Nk = floor(N/K);
    
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
end

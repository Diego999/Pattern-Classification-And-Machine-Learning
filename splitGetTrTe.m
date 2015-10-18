% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [XTr, yTr, XTe, yTe] = splitGetTrTe(y, X, idxCV, k)
    % get k'th subgroup in test, others in train
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    
    yTe = y(idxTe);
    XTe = X(idxTe,:);
    yTr = y(idxTr);
    XTr = X(idxTr,:);
end

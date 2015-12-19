% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [TTr, TTe] = splitGetTrTe(Tr, idxCV, k, isHog)
    % get k'th subgroup in test, others in train
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    
    if isHog
        TTr.X = Tr.X(idxTr, :);
        TTr.nX = Tr.nX(idxTr, :);
    end
    TTr.Z = Tr.Z(idxTr, :);
    TTr.nZ = Tr.nZ(idxTr, :);
    TTr.y = Tr.y(idxTr, :);
    
    if isHog
        TTe.X = Tr.X(idxTe, :);
        TTe.nX = Tr.nX(idxTe, :);
    end
    TTe.Z = Tr.Z(idxTe, :);
    TTe.nZ = Tr.nZ(idxTe, :);
    TTe.y = Tr.y(idxTe, :);
end

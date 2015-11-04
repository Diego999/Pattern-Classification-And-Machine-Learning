% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [mean_err_tr, mean_err_te] = runTestConstant(K, y, X)
    N = length(y);
    idxCV = splitGetCV(K, N);
    m = mean(y);
    
    for k=1:1:K
        [XTr, yTr, XTe, yTe] = splitGetTrTe(y, X, idxCV, k);

        tXTr = [ones(length(yTr),1) XTr];
        tXTe = [ones(length(yTe),1) XTe];

        err_tr(k) = RMSE(yTr, m);
        err_te(k) = RMSE(yTe, m);
    end
    
    mean_err_tr = mean(err_tr);
    mean_err_te = mean(err_te);
end
% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [mean_err_tr_ls, mean_err_te_ls, mean_err_tr_lsgd, mean_err_te_lsgd, mean_err_tr_rr, mean_err_te_rr] = runTest(K, y, X, alpha, lambda)
    N = length(y);
    idxCV = splitGetCV(K, N);

    for k=1:1:K
        [XTr, yTr, XTe, yTe] = splitGetTrTe(y, X, idxCV, k);

        tXTr = [ones(length(yTr),1) XTr];
        tXTe = [ones(length(yTe),1) XTe];

        beta_ls = leastSquares(yTr, tXTr);
        err_tr_ls(k) = RMSE(yTr, tXTr*beta_ls);
        err_te_ls(k) = RMSE(yTe, tXTe*beta_ls);

        beta_lsgd = leastSquaresGD(yTr, tXTr, alpha);
        err_tr_lsgd(k) = RMSE(yTr, tXTr*beta_lsgd);
        err_te_lsgd(k) = RMSE(yTe, tXTe*beta_lsgd);

        beta_rr = ridgeRegression(yTr, tXTr, lambda);
        err_tr_rr(k) = RMSE(yTr, tXTr*beta_rr);
        err_te_rr(k) = RMSE(yTe, tXTe*beta_rr);
    end
    
    mean_err_tr_ls = mean(err_tr_ls);
    mean_err_te_ls = mean(err_te_ls);
    
    mean_err_tr_lsgd = mean(err_tr_lsgd);
    mean_err_te_lsgd = mean(err_te_lsgd);
    
    mean_err_tr_rr = mean(err_tr_rr);
    mean_err_te_rr = mean(err_te_rr);
end
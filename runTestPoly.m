% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [mean_err_tr, mean_err_te] = runTestPoly(K, y, X, lambda)
    N = length(y);
    idxCV = splitGetCV(K, N);
    lambdas = logspace(-5,5,1000);
    degrees = [1 2];
    figure
    subplot(1,length(degrees),1);
    for d=1:1:length(degrees)
        degree = degrees(d);
        for k=1:1:K
            [XTr, yTr, XTe, yTe] = splitGetTrTe(y, X, idxCV, k);

            tXTr = [ones(length(yTr),1) buildPoly(XTr, degree)];
            tXTe = [ones(length(yTe),1) buildPoly(XTe, degree)];

            beta_rr = ridgeRegression(yTr, tXTr, lambda);
            err_tr_rr(k) = RMSE(yTr, tXTr*beta_rr);
            err_te_rr(k) = RMSE(yTe, tXTe*beta_rr);
        end

        mean_err_tr = mean(err_tr_rr);
        mean_err_te = mean(err_te_rr);

        subplot(1,length(degrees),d);
        semilogx(lambdas, mean_err_tr, 'b-o', lambdas, mean_err_te, 'r-x');
        hold on;
        legend('Train error', 'Test error', 'Location', 'southeast');
        xlabel(sprintf('RR lambda (degree %d)', degree));
        ylabel('error');
        axis([lambdas(1) lambdas(end) min([mean_err_tr mean_err_te])-10 max([mean_err_tr mean_err_te])+10]);
        hold off;
    end
end
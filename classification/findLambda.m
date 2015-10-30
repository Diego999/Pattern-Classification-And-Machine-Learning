% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

%idxCls1 is to indicate which cluster do we use (1 2 3). 0 means all the
%dataset
function [lambda] = findLambda(K, y_, X_, d1)
    % If we choose a cluster, we check if all the degrees were specified
    % Otherwise we choose arbitrary degree to 1 for each cluster

    if ~exist('d1', 'var')
        [y, X] = preprocess(y_, X_);  
    else
        [y, X] = preprocess(y_, X_, d1);
    end

    N = length(y);
    idxCV = splitGetCV(K, N);

    % K-fold
    for k=1:1:K
        [XTr, yTr, XTe, yTe] = splitGetTrTe(y, X, idxCV, k);

        tXTr = [ones(length(yTr),1) XTr];
        tXTe = [ones(length(yTe),1) XTe];

        lambdas = logspace(-2,2,50);
        for i = 1:1:length(lambdas)
            lambda = lambdas(i);
            beta_rr = penLogisticRegression(yTr, tXTr, 0.001, lambda);
            err_tr_rr(k,i) = RMSE(yTr, tXTr*beta_rr);
            err_te_rr(k,i) = RMSE(yTe, tXTe*beta_rr);
        end
    end

    mseTr = mean(err_tr_rr);
    mseTe = mean(err_te_rr);

    [errStar, lambdaIdStar] = min(mseTe);
    lambdaStar = lambdas(lambdaIdStar);
    
    figure
    semilogx(lambdas, mseTr, 'b-o', lambdas, mseTe, 'r-x');
    hold on;
    legend('Train error', 'Test error', 'Location', 'southeast');
    semilogx(lambdaStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    xlabel('lambda');
    ylabel('error');
    axis([lambdas(1) lambdas(end) min([mseTr mseTe])-50 max([mseTr mseTe])+50]);
    hold off;
    
    lambda = lambdaStar;
end

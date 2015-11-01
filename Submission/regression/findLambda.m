% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

%idxCls1 is to indicate which cluster do we use (1 2 3). 0 means all the
%dataset
function [lambda] = findLambda(K, y_, X_, idxCls, d1, d2, d3)
    % If we choose a cluster, we check if all the degrees were specified
    % Otherwise we choose arbitrary degree to 1 for each cluster
    if idxCls ~= 0
        if ~exist('d1', 'var') || ~exist('d2', 'var') || ~exist('d3', 'var')
            [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3] = preprocess(y_, X_);  
        else
            [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3] = preprocess(y_, X_, d1, d2, d3);
        end
    end
    
    % Select cluster if specified or the whole dataset
    if idxCls == 1
        y = y_cls1;
        X = X_cls1;
    elseif idxCls == 2
        y = y_cls2;
        X = X_cls2;
    elseif idxCls == 3
        y = y_cls3;
        X = X_cls3;
    else
        y = y_;
        X = X_;
    end

    N = length(y);
    idxCV = splitGetCV(K, N);

    % K-fold
    for k=1:1:K
        [XTr, yTr, XTe, yTe] = splitGetTrTe(y, X, idxCV, k);

        tXTr = [ones(length(yTr),1) XTr];
        tXTe = [ones(length(yTe),1) XTe];

        lambdas = logspace(-5,5,500);
        for i = 1:1:length(lambdas)
            lambda = lambdas(i);
            beta_rr = ridgeRegression(yTr, tXTr, lambda);
            err_tr_rr(k,i) = RMSE(yTr, tXTr*beta_rr);
            err_te_rr(k,i) = RMSE(yTe, tXTe*beta_rr);
        end
    end

    mseTr = mean(err_tr_rr);
    mseTe = mean(err_te_rr);

    [errStar, lambdaIdStar] = min(mseTe);
    lambdaStar = lambdas(lambdaIdStar);
    
%     figure
%     semilogx(lambdas, mseTr, 'b-o', lambdas, mseTe, 'r-x');
%     hold on;
%     legend('Train error', 'Test error', 'Location', 'southeast');
%     semilogx(lambdaStar, errStar, 'black-diamond', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
%     xlabel('lambda');
%     ylabel('error');
%     axis([lambdas(1) lambdas(end) min([mseTr mseTe])-50 max([mseTr mseTe])+50]);
%     hold off;
    
    lambda = lambdaStar;
end

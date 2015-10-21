% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [lambda] = findLambda(K, y_, X_, idxCls, d)
    [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3] = preprocess(y_, X_, d);
    if idxCls == 1
        y = y_cls1;
        X = X_cls1;
    elseif idxCls == 2
        y = y_cls2;
        X = X_cls2;
    else
        y = y_cls3;
        X = X_cls3;
    end

    N = length(y);
    idxCV = splitGetCV(K, N);

    for k=1:1:K
        [XTr, yTr, XTe, yTe] = splitGetTrTe(y, X, idxCV, k);

        tXTr = [ones(length(yTr),1) XTr];
        tXTe = [ones(length(yTe),1) XTe];

        lambdas = logspace(-5,5,1000);
        for i = 1:1:length(lambdas)
            lambda = lambdas(i);
            beta_rr = ridgeRegression(yTr, tXTr, lambda);
            err_tr_rr(k,i) = RMSE(yTr, tXTr*beta_rr);
            err_te_rr(k,i) = RMSE(yTe, tXTe*beta_rr);
        end
    end

    figure
    mseTr = mean(err_tr_rr);
    mseTe = mean(err_te_rr);

    [errStar, lambdaIdStar] = min(mseTe);
    lambdaStar = lambdas(lambdaIdStar);

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

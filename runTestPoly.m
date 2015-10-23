% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [mean_err_tr, mean_err_te] = runTestPoly(K, y_, X_, lambda, idxCls)
    degrees = [1 2];
    
    for d=1:1:length(degrees)
        degree = degrees(d);
        [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3] = preprocess(y_, X_, degree, degree, degree);
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

            beta_rr = ridgeRegression(yTr, tXTr, lambda);
            err_tr_rr(k) = RMSE(yTr, tXTr*beta_rr);
            err_te_rr(k) = RMSE(yTe, tXTe*beta_rr);
        end

        mean_err_tr(d) = mean(err_tr_rr);
        mean_err_te(d) = mean(err_te_rr);
    end
    
    figure
    semilogx(degrees, mean_err_tr, 'b-o', degrees, mean_err_te, 'r-x');
    hold on;
    legend('Train error', 'Test error', 'Location', 'southeast');
    xlabel('Degree');
    ylabel('error');
    axis([degrees(1) degrees(end) min([mean_err_tr mean_err_te])-10 max([mean_err_tr mean_err_te])+10]);
    hold off;
    
    mean_err_tr = mean_err_tr(end);
    mean_err_te = mean_err_te(end);
end
% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [mean_err_tr, mean_err_te] = runTestPoly(K, y_, X_, lambda, idxCls)
    degrees = [1 2 3 4 5];
    
    for s = 1:1:50
        setSeed(s);
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

            [XTr, yTr, XTe, yTe] = splitProp(0.8, y, X);
                

                tXTr = [ones(length(yTr),1) XTr];
                tXTe = [ones(length(yTe),1) XTe];

                beta_rr = ridgeRegression(yTr, tXTr, lambda);
                err_tr_rr(s, d) = RMSE(yTr, tXTr*beta_rr);
                err_te_rr(s, d) = RMSE(yTe, tXTe*beta_rr);
        end
    end
    mean_err_tr = mean(err_tr_rr);
    mean_err_te = mean(err_te_rr);
        
    figure
    
    plot(degrees, mean_err_tr,'b-','linewidth', 3);
    hold on;
    plot(degrees, mean_err_te,'r-','linewidth', 3);
    hold on;
    plot(degrees, err_te_rr, 'r-', 'color',[1 0.7 0.7]);
    hold on;
    plot(degrees, err_tr_rr, 'b-','color',[0.7 0.7 1]);
    hold off;
    legend('Train error', 'Test error', 'Location', 'northwest');
    xlabel('Degree');
    ylabel('RMSE');
    print('report/figures/degrees_cls1','-djpeg','-noui')
    
    mean_err_tr = mean_err_tr(end);
    mean_err_te = mean_err_te(end);
end


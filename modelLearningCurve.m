% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [err_tr, err_te_cv] = modelLearningCurve(prop, K, d, lambda, y, X)
    for i = 1:1:10
        setSeed(28111993*i);
        [XTr, yTr, XTe, yTe] = splitProp(prop, y, X);

        % CV Error
        N = length(yTr);
        idxCV = splitGetCV(K, N);
        for k=1:1:K
            [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);

            tXTr = [ones(length(yyTr),1) buildPoly(XXTr, d)];
            tXTe = [ones(length(yyTe),1) buildPoly(XXTe, d)];

            beta_rr = ridgeRegression(yyTr, tXTr, lambda);
            err_te_rr_cv(k) = RMSE(yyTe, tXTe*beta_rr);
        end
        err_te_cv(i) = mean(err_te_rr_cv);
        
        % Training Error
        tXTr = [ones(length(yTr),1) buildPoly(XTr, d)];
        beta_rr = ridgeRegression(yTr, tXTr, lambda);
        err_tr(i) = RMSE(yTr, tXTr*beta_rr);
    end
end

% % % Written by Diego Antognini & Jason Racine, EPFL 2015
% % % all rights reserved
% % 
% function [err_tr, err_te_cv] = modelLearningCurve(prop, K, d, lambda, y, X)
%     % Fix the test size
%     setSeed(1111);
%     [X_, y_, X_test, y_test] = splitProp(0.8, y, X);
%     
%     for i = 1:1:10
%         setSeed(28111993*i);
%         [XTr, yTr, XTe, yTe] = splitProp(prop, y_, X_);
% 
%         % CV Error
%         N = length(yTr);
%         idxCV = splitGetCV(K, N);
%         for k=1:1:K
%             [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
% 
%             tXTr = [ones(length(yyTr),1) XXTr];
%             tXTe = [ones(length(y_test),1) X_test];
% 
%             beta_rr = ridgeRegression(yyTr, tXTr, lambda);
%             err_te_rr_cv(k) = RMSE(y_test, tXTe*beta_rr);
%         end
%         err_te_cv(i) = mean(err_te_rr_cv);
%         
%         % Training Error
%         tXTr = [ones(length(yTr),1) buildPoly(XTr, d)];
%         beta_rr = ridgeRegression(yTr, tXTr, lambda);
%         err_tr(i) = RMSE(yTr, tXTr*beta_rr);
%     end
% end


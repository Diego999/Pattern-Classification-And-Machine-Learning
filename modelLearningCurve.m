% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [err_tr_cls1, err_te_cls1, err_tr_cls2, err_te_cls2, err_tr_cls3, err_te_cls3] = modelLearningCurve(prop, k1, k2, k3, y, X, d1, d2, d3, globalYTe, globalXTe)
    [XTr, yTr, XTe, yTe] = splitProp(prop, y, X);
    
     lambda_cls1 = findLambda(k1, yTr, XTr, 1, d1, d2, d3);
     lambda_cls2 = findLambda(k2, yTr, XTr, 2, d1, d2, d3);
     lambda_cls3 = findLambda(k3, yTr, XTr, 3, d1, d2, d3);
    
     [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(yTr, XTr, d1, d2, d3);

     tX_cls1 = [ones(length(y_cls1),1) X_cls1];
     tX_cls2 = [ones(length(y_cls2),1) X_cls2];
     tX_cls3 = [ones(length(y_cls3),1) X_cls3];
    
     beta_cls1 = ridgeRegression(y_cls1, tX_cls1, lambda_cls1);
     beta_cls2 = ridgeRegression(y_cls2, tX_cls2, lambda_cls2);
     beta_cls3 = ridgeRegression(y_cls3, tX_cls3, lambda_cls3);
     
     err_tr_cls1 = RMSE(y_cls1, tX_cls1*beta_cls1);
     err_tr_cls2 = RMSE(y_cls2, tX_cls2*beta_cls2);
     err_tr_cls3 = RMSE(y_cls3, tX_cls3*beta_cls3);
     
     [y_cls1, X_cls1, y_cls2, X_cls2, y_cls3, X_cls3, idx_cls1, idx_cls2, idx_cls3] = preprocess(globalYTe, globalXTe, d1, d2, d3);
 
     tXTe_cls1 = [ones(length(y_cls1),1) X_cls1];
     tXTe_cls2 = [ones(length(y_cls2),1) X_cls2];
     tXTe_cls3 = [ones(length(y_cls3),1) X_cls3];
     
     err_te_cls1 = RMSE(y_cls1, tXTe_cls1*beta_cls1);
     err_te_cls2 = RMSE(y_cls2, tXTe_cls2*beta_cls2);
     err_te_cls3 = RMSE(y_cls3, tXTe_cls3*beta_cls3);
end


% function [err_tr, err_te_cv] = modelLearningCurve(prop, K, d, lambda, y, X)
%     for i = 1:1:10
%         setSeed(28111993*i);
%         
% 
%         % CV Error
%         
%         for k=1:1:K
%             [XXTr, yyTr, XXTe, yyTe] = splitGetTrTe(yTr, XTr, idxCV, k);
% 
%             tXTr = [ones(length(yyTr),1) buildPoly(XXTr, d)];
%             tXTe = [ones(length(yyTe),1) buildPoly(XXTe, d)];
% 
%             beta_rr = ridgeRegression(yyTr, tXTr, lambda);
%             err_te_rr_cv(k) = RMSE(yyTe, tXTe*beta_rr);
%         end
%         err_te_cv(i) = mean(err_te_rr_cv);
%         
%         % Training Error
%         tXTr = [ones(length(yTr),1) buildPoly(XTr, d)];
%         beta_rr = ridgeRegression(yTr, tXTr, lambda);
%         err_tr(i) = RMSE(yTr, tXTr*beta_rr);
%     end
% end
% 
% 

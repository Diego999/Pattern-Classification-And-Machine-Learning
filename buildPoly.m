% % Written by Diego Antognini & Jason Racine, EPFL 2015
% % all rights reserved
% 
% function [X] = buildPoly(X_, degree)
%     for k = 1:degree
%         X(:,k) = X_.^k;
%     end
% end

function Xpoly = buildPoly(X,degree)
    Xpoly = X.^degree;
end


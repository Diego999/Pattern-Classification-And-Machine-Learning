% % Written by Diego Antognini & Jason Racine, EPFL 2015
% % all rights reserved
% 
% function [X] = buildPoly(X_, degree)
%     for k = 1:degree
%         X(:,k) = X_.^k;
%     end
% end

function Xpoly = buildPoly(X,degree)
    Xpoly = zeros(size(X,1), size(X,2)*degree);
    for k = 1:degree
        Xpoly(:,(k-1)*size(X,2)+1:1:k*size(X,2)) = X.^k;
    end
end


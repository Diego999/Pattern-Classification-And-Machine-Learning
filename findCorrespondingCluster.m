% x is a data vector. Both have to be normalized
function [idx_cls] = findCorrespondingCluster(x)
    if x(2) < 0.42
       idx_cls = 1;
    elseif x(16) < 1.17
       idx_cls = 2;
    else
       idx_cls = 3;
    end
end
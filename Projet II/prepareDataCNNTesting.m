function[Te] = prepareDataCNNTesting(Te_, M)
    [Te] = PCA(Te_, M);
end

function [Te] = PCA(Te_, M)
    Te = Te_;
    
    X = Te.nX;
    N = size(X, 1);
    D = size(X, 2);

    % Use method in Bishop 12.1.4
    Xu = mean(X, 1);
    
    % Transform to non-space matrix
    X2 = X - ones(N, 1) * Xu;
    
    % Compute S term
    S = X2 * X2' / N;
    
    % Compute and sort eigen values
    [V, l] = eig(S, 'vector');
    [l, I] = sort(l, 1, 'descend');
    V = V(:, I);
    l = abs(l); % To avoid neg values, only for the "0" which can be negative
    
    % Compute U term
    norm = sqrt(N*l)';
    norm = repmat(norm, D, 1);
    U = X2' * V ./ norm;
    
    Te.nZ = Te.nX * U(:, 1:M);
    Te.Z = Te.X * U(:, 1:M);
end

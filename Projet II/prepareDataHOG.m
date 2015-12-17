function[Tr, Te] = prepareDataHOG(Tr_, Te_, M)
    [Tr, Te] = PCA(Tr_, Te_, M);
end

function [Tr, Te] = PCA(Tr_, Te_, M)
    Tr = Tr_;
    Te = Te_;
    
    X = Tr.nX;
    N = size(X, 1);
    D = size(X, 2);
    S = cov(X, 1);
    
    % Manually sort eigenvectors
    [U, l] = eig(S, 'vector'); % eigenvector and eigenvalues
    [l, I] = sort(l, 1, 'descend');
    U = U(:, I);

    %J = computeDistortionMatrix(l, D);
    %displayDistortionMatrix(J);
    
    Tr.nZ = Tr.nX * U(:, 1:M);
    Te.nZ = Te.nX * U(:, 1:M);
    
    Tr.nZU = Tr.nX * U(:, 1:M) * U(:, 1:M)';
    Te.nZU = Te.nX * U(:, 1:M) * U(:, 1:M)';
end

function [J] = computeDistortionMatrix(l, D)
    % Here D < N
    J = zeros(D, 1);
    for m = 1:D
        J(m) = sum(l(m+1:D));
    end
end

function [] = displayDistortionMatrix(J)
    figure('Name', 'Distortion Measure');
    plot(J, 'LineWidth', 4);
    xlabel('M');
    ylabel('J');
    title('distortion J vs M');
end
% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

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
    
    Tr.Z = Tr.X * U(:, 1:M);
    Te.Z = Te.X * U(:, 1:M);
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
    ylim([0 5e3]);
    xlim([0 N]);
    xlabel('M');
    ylabel('J');
    title('distortion J vs M');
   
    set(gca,'fontsize', 18);
    set(gca,'LineWidth',2);
    SP=1.17;
    line([SP SP], [0 5e3],'Color',[0 0 0], 'LineWidth', 2);
    legend('Distortion');
    hold off;
    print('./report/figures/distortionHOG.jpg','-djpeg','-noui');
end
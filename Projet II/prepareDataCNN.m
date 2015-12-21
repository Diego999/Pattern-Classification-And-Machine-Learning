% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function[Tr, Te] = prepareDataCNN(Tr_, Te_, M)
    [Tr, Te] = PCA(Tr_, Te_, M);
end

function [Tr, Te] = PCA(Tr_, Te_, M)
    Tr = Tr_;
    Te = Te_;
    
    X = Tr.nX;
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
    
    %J = computeDistortionMatrix(l, N);
    %displayDistortionMatrix(J);
    
    Tr.nZ = Tr.nX * U(:, 1:M);
    Te.nZ = Te.nX * U(:, 1:M);
    
    Tr.Z = Tr.X * U(:, 1:M);
    Te.Z = Te.X * U(:, 1:M);
end

function [J] = computeDistortionMatrix(l, N)
    % We use N because N < D
    J = zeros(N, 1);
    for m = 1:N
        J(m) = sum(l(m+1:N));
    end
end

function [] = displayDistortionMatrix(J)
    figure('Name', 'Distortion Measure');
    plot(J, 'LineWidth', 4);
    ylim([0 4e4]);
    xlim([0 size(J,1)]);
    xlabel('M');
    ylabel('J');
    title('distortion J vs M');
   
    set(gca,'fontsize', 18);
    set(gca,'LineWidth',2);
    SP=1.17;
    line([SP SP], [0 4e4],'Color',[0 0 0], 'LineWidth', 2);
    legend('Distortion');
    hold off;
    % saveas(f2, 'plots/cluster/feature16.jpg');
    print('./report/figures/distortionCNN.jpg','-djpeg','-noui');

end
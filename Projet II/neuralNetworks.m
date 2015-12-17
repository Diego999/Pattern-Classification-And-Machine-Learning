function [err, nnPred] = neuralNetworks(Xtr, ytr, Xte, yTe, inputSize, innerSize, numepochs, batchsize, learningRate, binary)
    if binary
        outputSize = 2;
    else
        outputSize = 4;
    end
    
    nn = nnsetup([inputSize innerSize outputSize]);

    opts.numepochs  = numepochs;
    opts.batchsize  = batchsize; % http://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
    opts.plot       = 0;
    nn.learningRate = learningRate;

    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor(size(Xtr) / opts.batchsize);
    X = Xtr(1:numSampToUse, :);
    labels = ytr(1:numSampToUse);
    
    % prepare labels for NN
    if binary
        LL = [1 * (labels == 1), ... % first column, p(y=1)
          1 * (labels == 2)];        
    else
        LL = [1 * (labels == 1), ... % first column, p(y=1)
          1 * (labels == 2), ... % second column, p(y=2), etc
          1 * (labels == 3), ...
          1 * (labels == 4) ];
    end
    
    tstart = tic; % because nntrain uses tic too...
    [nn, ~] = nntrain(nn, X, LL, opts);
    toc(tstart)

    % to get the scores we need to do nnff (feed-forward)
    tstart = tic;
    nn.testing = 1;
    nn = nnff(nn, Xte, zeros(size(Xte, 1), nn.size(end)));
    nn.testing = 0;

    % predict on the test set
    nnPred = nn.a{end};

    % get the most likely class
    [~, predictions] = max(nnPred, [], 2);
    toc(tstart)
    
    err = balancedErrorRate(yTe, predictions);
end


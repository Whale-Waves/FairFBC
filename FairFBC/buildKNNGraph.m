function S = buildKNNGraph(X, k, opts)

if nargin < 3, opts = struct(); end
if ~isfield(opts,'distance'), opts.distance = 'euclidean'; end
if ~isfield(opts,'symmetrizeMethod'), opts.symmetrizeMethod = 'max'; end

[n, ~] = size(X);
k = min(k, n-1);
% Use knnsearch to get indices and squared distances
% We query self + k neighbors -> K = k+1, then drop self.
K = k + 1;
[idx, dists] = knnsearch(X, X, 'K', K, 'Distance', opts.distance);

% Remove self (first column)
idx = idx(:, 2:end);
dists = dists(:, 2:end);

% Estimate sigma by median of distances to k-th neighbor
kth_dists = dists(:, end);
sigma = median(kth_dists);
if sigma <= 0
    sigma = mean(kth_dists(kth_dists>0)) + eps;
end
sigma = max(sigma, eps);

% Gaussian kernel weights
weights = exp(-(dists.^2) / (sigma^2 + eps));

% Build sparse directed adjacency (i -> neighbor)
I = repmat((1:n)', 1, k);
I = I(:);
J = idx(:);
V = weights(:);

S = sparse(I, J, V, n, n);

% Symmetrize
switch lower(opts.symmetrizeMethod)
    case 'max'
        S = max(S, S');
    case 'avg'
        S = (S + S') * 0.5;
    otherwise
        S = max(S, S');
end

% Ensure no isolated nodes have zero degree - keep as is; APPR handles deg==0
end

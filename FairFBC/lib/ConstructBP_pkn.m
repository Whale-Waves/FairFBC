function Z = ConstructBP_pkn(X, anchors, varargin)
% Input
%         X: nSmp * nFea
%         anchors: nAnchor * nFea
%         nNeighbor: row sparsity
% Output
%         Z: nSmp * nAnchor
%
% [1] Clustering and Projected Clustering with Adaptive Neighbors, KDD,
% 2014
%

[nSmp, nFea] = size(X);
[nAnchor, nFea] = size(anchors);

param_names = {'nNeighbor'};
param_default =  {7};
[eid, errmsg, nNeighbor] = getargs(param_names, param_default, varargin{:});
if ~isempty(eid)
    error(sprintf('ConstructBP_pkn:%s', eid), errmsg);
end

% Use NaN-safe distance if data contains missing values; otherwise fall back to original
if any(isnan(X(:))) || any(isnan(anchors(:)))
    % NaN-safe squared Euclidean distance with valid-dimension scaling
    Xz = X; Xz(isnan(Xz)) = 0;
    Az = anchors; Az(isnan(Az)) = 0;
    maskX = ~isnan(X);
    maskA = ~isnan(anchors);
    % Overlap counts per pair (nSmp x nAnchor)
    C = maskX * maskA';
    % Sums over overlapping dimensions
    Xsq = Xz.^2; Asq = Az.^2;
    Sx = Xsq * maskA';                 % sum of x^2 over dims where anchor is valid
    Sa = maskX * Asq';                 % sum of a^2 over dims where x is valid
    cross = Xz * Az';                  % sum of x*a over dims where both are valid
    D = Sx + Sa - 2 * cross;           % squared distance over overlapping dims
    D(D < 0) = 0;
    % Scale to account for varying number of valid dims
    scale = nFea ./ max(C, 1);
    D = D .* scale;
    % If no overlapping dims, set a large finite distance to avoid selection
    D(C == 0) = 1e12;
else
    D = EuDist2(X, anchors, 0); % O(nmd)
    D (D<0) = 0;
end
[D2, Idx] = sort(D, 2); % sort each row
v1 = D2(:, nNeighbor+1);
v2 = D2(:, 1:nNeighbor);
v3 = bsxfun(@minus, v1, v2);
v4 = 1./max(nNeighbor * v1 - sum(v2, 2), eps);
v5 = bsxfun(@times, v3, v4);
row_idx = repmat((1:nSmp)', nNeighbor, 1);
idx_k = Idx(:, 1:nNeighbor);
Z = sparse(row_idx, idx_k(:), v5(:), nSmp, nAnchor, nSmp * nNeighbor);
end
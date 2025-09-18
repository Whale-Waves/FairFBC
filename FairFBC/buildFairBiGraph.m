function [B, A, anchor_ids] = buildFairBiGraph(X, Yg, m, knn_size, opts)

if nargin < 5, opts = struct(); end
if ~isfield(opts,'anchor_replicates'), opts.anchor_replicates = 5; end
if ~isfield(opts,'knn_graph_k'), opts.knn_graph_k = knn_size; end
if ~isfield(opts,'diff_alpha'), opts.diff_alpha = 0.15; end
if ~isfield(opts,'diff_eps'), opts.diff_eps = 1e-4; end
if ~isfield(opts,'diff_max_push'), opts.diff_max_push = 2e6; end
if ~isfield(opts,'useParallel'), opts.useParallel = true; end
if ~isfield(opts,'seed_k'), opts.seed_k = 1; end

% Validate inputs
[n, d] = size(X);
if size(Yg,1) ~= n, error('Yg must have same number of rows as X'); end
if m < 1, error('m must be >= 1'); end

% 1) Balanced anchor selection (keeps your fairness logic)
[A, anchor_ids, anchors_per_group] = selectBalancedAnchors(X, Yg, m, opts);

% 2) Build sample-sample kNN graph (symmetric weighted)
if isempty(opts.knn_graph_k)
    opts.knn_graph_k = max(20, min(50, ceil(0.02*n))); % reasonable default
end
S = buildKNNGraph(X, opts.knn_graph_k, opts);

% 3) Graph diffusion: compute APPR from each anchor seed (seed = anchor_ids)
diffOpts.alpha = opts.diff_alpha;
diffOpts.eps = opts.diff_eps;
diffOpts.max_push = opts.diff_max_push;
diffOpts.useParallel = opts.useParallel;
diffOpts.seed_k = opts.seed_k;

B = graphDiffusionConstruct(S, anchor_ids, diffOpts);

end

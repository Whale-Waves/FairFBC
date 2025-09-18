function B = graphDiffusionConstruct(S, anchor_ids, opts)

if nargin < 3, opts = struct(); end
if ~isfield(opts,'alpha'), opts.alpha = 0.15; end
if ~isfield(opts,'eps'), opts.eps = 1e-4; end
if ~isfield(opts,'max_push'), opts.max_push = 2e6; end
if ~isfield(opts,'useParallel'), opts.useParallel = true; end
if ~isfield(opts,'seed_k'), opts.seed_k = 1; end

[n, ~] = size(S);
anchor_ids = anchor_ids(:);
m = numel(anchor_ids);

% Precompute CSR-like structures for adjacency to speed up pushes
[i_idx, j_idx, v_idx] = find(S);
% Sort by row (i_idx)
[rows_sorted, perm] = sort(i_idx);
i_idx = i_idx(perm); j_idx = j_idx(perm); v_idx = v_idx(perm);
% row_ptr: start index in j_idx/v_idx for each row r is row_ptr(r), end exclusive row_ptr(r+1)
counts = accumarray(i_idx,1,[n,1]);
row_ptr = zeros(n+1,1);
row_ptr(1) = 1;
row_ptr(2:end) = 1 + cumsum(counts);
% neighbors and weights vector
neighbors = j_idx;
weights = v_idx;
% degree vector
deg = full(sum(S,2));
deg(deg==0) = 0; % keep zeros

% Pre-allocate sparse storage: we will collect nonzeros for each column
cols_i = cell(m,1);
cols_v = cell(m,1);


% Decide parallel or not
doPar = false;
if opts.useParallel
    try
        pool = gcp('nocreate');
        if isempty(pool)
            % try to open pool
            parpool('threads'); % fallback to threads-based if available
            pool = gcp('nocreate');
        end
        if ~isempty(pool)
            doPar = true;
        end
    catch
        doPar = false;
    end
end

% Extract scalar parameters to avoid broadcasting entire opts struct in parfor
alpha = opts.alpha;
eps_val = opts.eps;
max_push = opts.max_push;

if doPar
    
    parfor a = 1:m
        seed = anchor_ids(a);
        p = approxPPRPush(row_ptr, neighbors, weights, deg, seed, alpha, eps_val, max_push, n);
        if nnz(p) == 0
            cols_i{a} = []; cols_v{a} = [];
        else
            [ii, vv] = find(p);
            cols_i{a} = ii;
            cols_v{a} = vv;
        end
    end
else
    fprintf('graphDiffusionConstruct: running APPR for %d anchors (serial)\n', m);
    for a = 1:m
        seed = anchor_ids(a);
        p = approxPPRPush(row_ptr, neighbors, weights, deg, seed, alpha, eps_val, max_push, n);
        if nnz(p) == 0
            cols_i{a} = []; cols_v{a} = [];
        else
            [ii, vv] = find(p);
            cols_i{a} = ii;
            cols_v{a} = vv;
        end
    end
end

% Build sparse matrix B from cols_i/cols_v
% compute totals of nonzeros
nnz_total = 0;
for a = 1:m
    nnz_total = nnz_total + numel(cols_i{a});
end
ii_all = zeros(nnz_total,1);
jj_all = zeros(nnz_total,1);
vv_all = zeros(nnz_total,1);
pos = 1;
for a = 1:m
    if isempty(cols_i{a}), continue; end
    klen = numel(cols_i{a});
    idx_range = pos:(pos+klen-1);
    ii_all(idx_range) = cols_i{a};
    jj_all(idx_range) = a;
    vv_all(idx_range) = cols_v{a};
    pos = pos + klen;
end

B = sparse(ii_all, jj_all, vv_all, n, m);

% Row-normalize B (efficient sparse scaling)
row_sums = sum(B,2);
row_sums(row_sums==0) = 1;  % Prevent division by zero for isolated nodes
d = 1 ./ row_sums;          % Inverse row sums
[i,j,v] = find(B);          % Get non-zero triplets
v = v .* d(i);              % Scale values by corresponding row inverse
B = sparse(i,j,v,n,m);      % Reconstruct normalized sparse matrix

% Handle isolated nodes (rows that were originally zero)
zero_rows = find(sum(B,2) == 0);
if ~isempty(zero_rows)
    % Connect isolated samples uniformly to all anchors
    B(zero_rows, :) = repmat(1/m, numel(zero_rows), m);
end

end


%% approxPPRPush - local function implementing push algorithm
function p = approxPPRPush(row_ptr, neighbors, weights, deg, seed, alpha, eps, max_push, n)

% Preallocate
p_vec = sparse([],[],[], n,1);
r = sparse(seed, 1, 1, n, 1); % residual vector (sparse)
in_queue = false(n,1);

% init queue with seed if threshold (using MATLAB native array instead of Java)
if deg(seed) == 0
    threshold_seed = Inf;
else
    threshold_seed = r(seed) / deg(seed);
end

% Use pre-allocated MATLAB array as queue (compatible with parfor)
% For highly connected graphs, use generous queue size (64GB RAM can handle this)
max_queue_size = min(100*n, n*100);  % Very generous: 10x dataset size, capped at 1M
queue = zeros(max_queue_size, 1);
queue(1) = seed;
queue_head = 1;   % Index of next element to process
queue_tail = 1;   % Index of last element in queue
in_queue(seed) = true;

push_count = 0;
while queue_head <= queue_tail
    u = queue(queue_head);
    queue_head = queue_head + 1;
    in_queue(u) = false;
    
    ru = full(r(u));
    if ru == 0
        continue;
    end
    du = deg(u);
    % deposit to p
    addP = alpha * ru;
    if addP ~= 0
        p_vec(u) = p_vec(u) + addP;
    end
    remainder = (1 - alpha) * ru;
    r(u) = 0;
    push_count = push_count + 1;
    if push_count > max_push
        warning('approxPPRPush: reached max_push (%d); aborting early', max_push);
        break;
    end
    if du == 0
        % no neighbors: remainder disappears (or could be redistributed uniformly)
        continue;
    end
    % iterate neighbors: indices = row_ptr(u) : row_ptr(u+1)-1
    start_idx = row_ptr(u);
    end_idx = row_ptr(u+1)-1;
    if end_idx < start_idx
        continue;
    end
    neigh_inds = neighbors(start_idx:end_idx);
    neigh_w = weights(start_idx:end_idx);
    % distribute remainder proportionally
    % delta_v = remainder * w_uv / du
    delta = (remainder / du) * neigh_w; % vector
    % accumulate into r
    for t = 1:numel(neigh_inds)
        v = neigh_inds(t);
        inc = delta(t);
        if inc == 0, continue; end
        r(v) = r(v) + inc;
        % push threshold check
        if deg(v) > 0
            if full(r(v))/deg(v) > eps && ~in_queue(v)
                if queue_tail < max_queue_size
                    queue_tail = queue_tail + 1;
                    queue(queue_tail) = v;
                    in_queue(v) = true;
                else
                    % Queue overflow protection - rare but possible
                    warning('Queue overflow in APPR push, stopping early');
                    break;
                end
            end
        end
    end
end

% p_vec is sparse vector with approximate PPR mass; normalize to sum to 1 (optional)
ps = sum(p_vec);
if ps > 0
    p_vec = p_vec / ps;
end
p = p_vec;
end

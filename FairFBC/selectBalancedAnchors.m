function [A, anchor_ids, anchors_per_group] = selectBalancedAnchors(X, Yg, m, opts)

if nargin < 4, opts = struct(); end
if ~isfield(opts,'anchor_replicates'), opts.anchor_replicates = 5; end

[n, d] = size(X);
[~, G] = size(Yg);

gs = full(sum(Yg,1)); % 1 x G
non_empty_groups = find(gs > 0);
n_non_empty = numel(non_empty_groups);

if m < n_non_empty
    error('Requested m (%d) must be at least number of non-empty groups (%d)', m, n_non_empty);
end

% initial allocation: 1 per non-empty group
anchors_per_group = zeros(1, G);
anchors_per_group(non_empty_groups) = 1;
remaining = m - n_non_empty;

if remaining > 0
    % proportional allocation by group size
    gs_frac = gs / sum(gs);
    for ig = non_empty_groups
        additional = floor(remaining * gs_frac(ig));
        anchors_per_group(ig) = anchors_per_group(ig) + additional;
    end
    % distribute leftover due to floor rounding to largest groups
    total_assigned = sum(anchors_per_group(non_empty_groups));
    if total_assigned < m
        % find group(s) with largest size(s) among non-empty
        [~, idx_max] = sort(gs(non_empty_groups), 'descend');
        need_more = m - total_assigned;
        anchors_per_group(non_empty_groups(idx_max(1:need_more))) = ...
            anchors_per_group(non_empty_groups(idx_max(1:need_more))) + 1;
    end
end

% Collect anchors per group (capping at group size)
anchor_centers = cell(G,1);
anchor_ids = [];
A_list = [];
for g = 1:G
    if gs(g) == 0, continue; end
    num_anchors = anchors_per_group(g);
    idx_g = find(Yg(:,g) == 1);
    ng = numel(idx_g);
    num_anchors = min(num_anchors, ng); % cannot exceed number of samples in group
    if num_anchors == 0, continue; end
    % k-means within group using optimized lite_kmeans (2-10x faster)
    if num_anchors == 1
        % center = mean is reasonable; use mean of group
        C = mean(X(idx_g, :), 1);
    else
        % Use lite_kmeans for better performance than built-in kmeans
        try
            [~, C] = lite_kmeans(X(idx_g,:), num_anchors, 'Replicates', max(5,opts.anchor_replicates), 'MaxIter', 200);
        catch
            % fallback to kmeans++ style sampling
            rng(0,'twister');
            perm = randperm(ng, min(num_anchors, ng));
            C = X(idx_g(perm), :);
        end
    end
    % collect
    A_list = [A_list; C];
end

A = A_list;
m_actual = size(A,1);

% Map anchors to nearest sample indices (for seeding)
% Use knnsearch
anchor_ids = knnsearch(X, A, 'K', 1);

end

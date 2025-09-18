function label = fairness_aware_assignment(F, Yg, nCluster, opts)

if nargin < 4, opts = struct(); end
if ~isfield(opts,'ambig_quantile'), opts.ambig_quantile = 0.25; end
if ~isfield(opts,'max_exact_ambiguous'), opts.max_exact_ambiguous = 10000; end
if ~isfield(opts,'verbose'), opts.verbose = true; end

[n, c] = size(F);
if c ~= nCluster
    error('nCluster must equal number of columns in F');
end

% Basic data
[~, group] = max(Yg, [], 2);    % group labels 1..G
G = size(Yg,2);

% 1) Greedy baseline
[~, label_greedy] = max(F, [], 2);

% 2) Find ambiguous samples via adaptive gap threshold (max - 2nd max)
[sortedVals, ~] = sort(F, 2, 'descend');
gaps = sortedVals(:,1) - sortedVals(:,2);
thresh = quantile(gaps, opts.ambig_quantile);
ambig_mask = (gaps <= thresh);
ambig_idx = find(ambig_mask);
n_a = numel(ambig_idx);
if opts.verbose
    fprintf('Ambiguous threshold (quantile %.3g) = %.6g, ambiguous samples = %d / %d\n', ...
            opts.ambig_quantile, thresh, n_a, n);
end

% If no ambiguous, return greedy
if n_a == 0
    label = label_greedy;
        return;
    end
    
% 3) Compute remaining capacity rem_gk for ambiguous samples

group_counts = sum(Yg,1)';                  % G x 1 total group sizes
% current (non-ambiguous) assignment counts
nonamb_idx = find(~ambig_mask);
[grp_nonamb, lbl_nonamb] = deal(group(nonamb_idx), label_greedy(nonamb_idx));
curr_nonambig = zeros(G, c);
if ~isempty(nonamb_idx)
    curr_nonambig = full(accumarray([grp_nonamb, lbl_nonamb], 1, [G, c]));
end

% ambiguous counts per group
ambig_group_counts = sum(Yg(ambig_idx, :), 1)'; % G x 1

% desired per group per cluster (float)
desired_per_cluster = (group_counts / c);         % G x 1, same for each cluster
desired_matrix = repmat(desired_per_cluster, 1, c); % G x c

raw_rem_real = desired_matrix - curr_nonambig;
raw_rem_real(raw_rem_real < 0) = 0;

% Convert raw_rem_real to integer rem_gk with constraints sum(rem_gk(g,:)) == ambiguous_group_counts(g)
rem_gk = zeros(G, c);
for g = 1:G
    need = ambig_group_counts(g);
    row_real = raw_rem_real(g, :);
    if need <= 0
        rem_gk(g, :) = 0;
        continue;
    end
    % If all zeros or sum(row_real) == 0, spread uniformly
    if sum(row_real) <= 1e-12
        base = floor(need / c) * ones(1,c);
        leftover = need - sum(base);
        if leftover > 0
            base(1:leftover) = base(1:leftover) + 1;
        end
        rem_gk(g, :) = base;
        continue;
    end
    % Otherwise do floor + distribute fractional parts
    base = floor(row_real);
    frac = row_real - base;
    base_sum = sum(base);
    leftover = need - base_sum;
    % If leftover negative (shouldn't happen because raw_rem_real >=0), clamp
    if leftover <= 0
        % If base_sum >= need: take top 'need' bins by base, but base are integers
        % a safe fallback: greedy remove from largest base until sum==need
        % (unlikely), implement robustly:
        base_sorted = sort(base,'descend');
        % remove difference
        diffrem = base_sum - need;
        if diffrem > 0
            % subtract from largest buckets
            [~, order_desc] = sort(base,'descend');
            idxr = 1;
            while diffrem > 0 && idxr <= c
                take = min(base(order_desc(idxr)), diffrem);
                base(order_desc(idxr)) = base(order_desc(idxr)) - take;
                diffrem = diffrem - take;
                idxr = idxr + 1;
            end
        end
        rem_gk(g, :) = base;
    else
        % distribute leftover according to largest fractional parts
        [~, ord_frac] = sort(frac, 'descend');
        add = zeros(1,c);
        add(ord_frac(1:leftover)) = 1;
        rem_gk(g, :) = base + add;
    end
end

% Sanity: sums must match ambiguous per group
diffs = sum(rem_gk,2) - ambig_group_counts;
if any(abs(diffs) > 0)
    % Small numerical fix: adjust
    for g = 1:G
        d = sum(rem_gk(g,:)) - ambig_group_counts(g);
        if d > 0
            % remove from largest entries
            [~,ord] = sort(rem_gk(g,:),'descend');
            idx = 1;
            while d>0 && idx<=c
                can = min(rem_gk(g,ord(idx)), d);
                rem_gk(g,ord(idx)) = rem_gk(g,ord(idx)) - can;
                d = d - can; idx = idx+1;
            end
        elseif d < 0
            % add to largest fractional from raw_rem_real
            [~,ord] = sort(raw_rem_real(g,:)-floor(raw_rem_real(g,:)),'descend');
            idx = 1;
            while d<0 && idx<=c
                rem_gk(g,ord(idx)) = rem_gk(g,ord(idx)) + 1;
                d = d + 1; idx = idx+1;
            end
        end
    end
end

% Clip negatives, ensure integer
rem_gk(rem_gk < 0) = 0;
rem_gk = round(rem_gk);

% Final check
if any(sum(rem_gk,2) ~= ambig_group_counts)
    % As last resort, set distribution proportional to group desired
    for g = 1:G
        need = ambig_group_counts(g);
        if need == 0, continue; end
        v = raw_rem_real(g,:);
        if sum(v) < 1e-12
            v = ones(1,c);
        end
        v = v / sum(v);
        ints = floor(v * need);
        leftover = need - sum(ints);
        [~, ord] = sort(v - floor(v),'descend');
        % Safety check: ensure leftover doesn't exceed available clusters
        leftover = min(leftover, c);
        for k = 1:leftover, ints(ord(k)) = ints(ord(k))+1; end
        rem_gk(g,:) = ints;
    end
end

% 4) Build baseline label_final with greedy labels and replace ambiguous by solved ones
label_final = label_greedy;

% Attempt exact assignment if small enough
if n_a <= opts.max_exact_ambiguous
    if opts.verbose
        fprintf('Attempting exact assignment on ambiguous subset (n_a=%d) ...\n', n_a);
    end
    try
        % Try to use integer assignment by expanding cluster slots (slot-level assignment)
        % Build "slots": for each group g and cluster k, create rem_gk(g,k) identical slots
        total_slots = sum(rem_gk(:));
        if total_slots ~= n_a
            error('slot count mismatch: total_slots=%d, n_a=%d', total_slots, n_a);
        end
        % Build cost matrix (n_a x n_a): row -> ambiguous sample, col -> slot
      
        try
            
            slot_cluster = zeros(total_slots,1);
            slot_group = zeros(total_slots,1);
            ptr = 0;
            for g = 1:G
                for k = 1:c
                    cnt = rem_gk(g,k);
                    if cnt > 0
                        slot_cluster(ptr + (1:cnt)) = k;
                        slot_group(ptr + (1:cnt)) = g;
                        ptr = ptr + cnt;
                    end
                end
            end
            
            Fsub = F(ambig_idx, :);  % n_a x c
            
            if exist('matchpairs','file') == 2
                % Build full cost matrix (may be large but n_a small here)
                costMat = zeros(n_a, total_slots);
                for j = 1:total_slots
                    k_j = slot_cluster(j);
                    costMat(:, j) = -Fsub(:, k_j);  % cost: -F (prefer larger F)
                end
                % matchpairs finds min cost matching
                [pairs, ~] = matchpairs(costMat, -Inf); % pairs: [row, col]
                % pairs may not include all rows if rectangular; ensure full coverage
                if size(pairs,1) < n_a
                    error('matchpairs returned partial matching');
                end
                assigned_rows = pairs(:,1);
                assigned_cols = pairs(:,2);
                % convert to labels
                for t = 1:length(assigned_rows)
                    rowIdx = assigned_rows(t);
                    colIdx = assigned_cols(t);
                    k_assigned = slot_cluster(colIdx);
                    global_i = ambig_idx(rowIdx);
                    label_final(global_i) = k_assigned;
                end
                label = label_final;
                return;
            else
                error('matchpairs not found or not reliable here');
            end
        catch % if exact slot-level fails, fallback to greedy global candidate assign below
            if opts.verbose
                fprintf('Exact slot-level assignment unavailable or failed -> fallback to greedy assignment.\n');
            end
        end
    catch ME
        if opts.verbose
            fprintf('Exact assignment attempt failed: %s\n', ME.message);
        end
        end
    end
    
% Otherwise (large or exact failed): greedy global candidate assignment
if opts.verbose
    fprintf('Running global greedy candidate assignment for ambiguous subset (n_a=%d, c=%d)...\n', n_a, c);
end

% Prepare per-ambiguous sample structures
Fsub = F(ambig_idx, :);               % n_a x c
group_sub = group(ambig_idx);         % n_a x 1

% Build candidate list: (i_local, k) pairs
% To reduce memory if n_a*c huge, we do chunked sorting: but here we assume n_a moderate.
ii = repmat((1:n_a)', 1, c);
kk = repmat(1:c, n_a, 1);
ii = ii(:);
kk = kk(:);
% Create linear indices safely
linear_indices = (kk - 1) * n_a + ii; 
scores = -Fsub(linear_indices);  

[sortedScores, ord] = sort(scores, 'ascend');
ii = ii(ord);
kk = kk(ord);

assigned_local = false(n_a,1);
rem_gk_local = rem_gk;  % G x c
assigned_count = 0;

for t = 1:length(ii)
    i_local = ii(t);
    k = kk(t);
    if assigned_local(i_local)
        continue;
    end
    g = group_sub(i_local);
    if rem_gk_local(g,k) > 0
        % assign
        global_i = ambig_idx(i_local);
        label_final(global_i) = k;
        assigned_local(i_local) = true;
        rem_gk_local(g,k) = rem_gk_local(g,k) - 1;
        assigned_count = assigned_count + 1;
        if assigned_count == n_a
            break;
        end
    end
end

% If still unassigned (rare), assign to best available cluster respecting any remaining capacity,
% otherwise fall back to argmax.
if any(~assigned_local)
    if opts.verbose
        fprintf('Greedy pass left %d unassigned; performing repair step...\n', sum(~assigned_local));
    end
    % For each unassigned sample, try to assign to cluster with rem > 0, prefer largest F
    un_idx_local = find(~assigned_local);
    for t = 1:length(un_idx_local)
        i_local = un_idx_local(t);
        global_i = ambig_idx(i_local);
        g = group_sub(i_local);
        % find clusters with capacity
        caps = find(rem_gk_local(g,:) > 0);
        if ~isempty(caps)
            % choose argmax among caps
            [~,bestkIdx] = max(Fsub(i_local, caps));
            k = caps(bestkIdx);
            label_final(global_i) = k;
            rem_gk_local(g,k) = rem_gk_local(g,k) - 1;
            assigned_local(i_local) = true;
        else
            % no remaining capacity (should not happen) -> assign greedy argmax
            [~,k] = max(Fsub(i_local,:));
            label_final(global_i) = k;
            assigned_local(i_local) = true;
        end
    end
end

label = label_final;
end



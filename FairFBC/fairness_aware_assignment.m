function label = fairness_aware_assignment(F, Yg, nCluster, opts)

if nargin < 4, opts = struct(); end
% Default quantile parameter tau from paper (default tau=0.10)
if ~isfield(opts,'ambig_quantile'), opts.ambig_quantile = 0.10; end
% max_exact_ambiguous can control when to fallback to greedy if matching is too slow
% Though the paper solves the matching problem exactly (Eq 18).
if ~isfield(opts,'max_exact_ambiguous'), opts.max_exact_ambiguous = 15000; end 
if ~isfield(opts,'verbose'), opts.verbose = true; end

[n, c] = size(F);
if c ~= nCluster
    error('nCluster must equal number of columns in F');
end

% Extract group indices
[~, group] = max(Yg, [], 2);    % group labels 1..G
G = size(Yg,2);

%% 1) Freeze confident assignments.
% Define greedy label l_i = argmax_k F_{ik}
[sortedF, sortedIdx] = sort(F, 2, 'descend');
label_greedy = sortedIdx(:, 1);

% Compute confidence gap Delta_i = sigma_i^(1) - sigma_i^(2)
gaps = sortedF(:, 1) - sortedF(:, 2);

% Define threshold t = Quantile_tau(Delta_i)
t = quantile(gaps, opts.ambig_quantile);

% Split into N (confident) and A (ambiguous) sets
% N = {i : Delta_i > t}, A = X \ N
confident_mask = (gaps > t);
ambig_mask = ~confident_mask;

confident_idx = find(confident_mask);
ambig_idx = find(ambig_mask);

% Initialize final labels with greedy labels
label = label_greedy;

% All i in N are fixed to y_hat_i = l_i (already set in 'label')

if isempty(ambig_idx)
    if opts.verbose
        fprintf('No ambiguous samples found. Returning greedy labels.\n');
    end
    return;
end

n_a = length(ambig_idx);
if opts.verbose
    fprintf('Ambiguous threshold (quantile %.3g) = %.6g, ambiguous samples = %d / %d\n', ...
            opts.ambig_quantile, t, n_a, n);
end


%% 2) Derive group--cluster quotas for ambiguous samples.
% C_{gk} = |{i in N : g(i)=g, y_hat_i=k}|
% a_g = |{i in A : g(i)=g}|

% Calculate C_{gk}
groups_conf = group(confident_idx);
labels_conf = label(confident_idx);

% Careful with sparse accumarray if confident_idx is empty
if isempty(confident_idx)
    C_gk = zeros(G, c);
else
    C_gk = full(accumarray([groups_conf, labels_conf], 1, [G, c]));
end

% Calculate a_g
a_g = histcounts(group(ambig_idx), 1:G+1)';

% Calculate group totals n_g
n_g = sum(Yg, 1)';

% Remaining quota R_real_{gk} = max(0, n_g/c - C_{gk})
target_per_cluster = n_g / c; % G x 1
target_matrix = repmat(target_per_cluster, 1, c); % G x c

R_real = max(0, target_matrix - C_gk);

% Convert R_real to integer vector R_{g,:} such that sum_k R_{gk} = a_g
R_int = zeros(G, c);

for g = 1:G
    ag_val = a_g(g);
    
    if ag_val == 0
        continue;
    end
    
    row_real = R_real(g, :);
    
    % If row_real is all zeros but we need to assign a_g samples (rare edge case where C_gk exceeded target everywhere),
    % we must distribute a_g somehow. Fallback to distributing based on targets or uniform.
    if sum(row_real) <= 1e-10
         % Fallback: distribute proportionally to original targets or just uniformly
         % Here we use a safe uniform-like distribution mechanism
         row_real = ones(1, c); 
    end
    
    % Standard largest remainder method / similar logic to ensure sum constraint
    % 1. Scale row_real so it sums to a_g
    row_scaled = row_real * (ag_val / sum(row_real));
    
    % 2. Take floor
    row_floor = floor(row_scaled);
    
    % 3. Compute remainder
    rem_count = ag_val - sum(row_floor);
    
    % 4. Distribute remainder to largest fractional parts
    fractional_part = row_scaled - row_floor;
    [~, sort_idx] = sort(fractional_part, 'descend');
    
    row_final = row_floor;
    if rem_count > 0
        % Distribute 1 to top 'rem_count' indices
        % Ensure we don't go out of bounds if rem_count > c (shouldn't happen with floor logic generally)
        for k_idx = 1:rem_count
             idx_to_inc = sort_idx(mod(k_idx-1, c) + 1);
             row_final(idx_to_inc) = row_final(idx_to_inc) + 1;
        end
    end
    
    R_int(g, :) = row_final;
end

%% 3) Fair matching on ambiguous samples (maximum weight).
% Maximize sum_{i in A} sum_k Z_{ik} F_{ik}
% s.t. sum_k Z_{ik} = 1
%      sum_{i in A_g} Z_{ik} = R_{gk}

% This is a maximum weight matching problem.
% Construct the cost matrix / bi-partite graph.
% We have n_a ambiguous samples.
% We have sum(R_int) slots. Note sum(R_int(:)) should equal n_a.
% Check sum constraint:
if sum(R_int(:)) ~= n_a
    % This might happen due to numerical issues or empty groups logic, 
    % generally with logical construction above it holds.
    % Force fix if needed or error.
    warning('Quota sum %d does not match ambiguous count %d. Adjusting last element.', sum(R_int(:)), n_a);
    diff_val = n_a - sum(R_int(:));
    % Simply add diff to the first non-zero slot found or just the first slot
    R_int(1,1) = max(0, R_int(1,1) + diff_val);
end

% We can use matchpairs (linear assignment problem solver) for exact solution.
% Cost matrix C for matchpairs: rows=samples, cols=slots.
% matchpairs minimizes cost, so we use -F_{ik}.
%
% Implementation Detail:
% Expanding R_int into individual slots can be very large if n_a is large.
% If n_a is very large, 'matchpairs' might be slow (O(N^3)).
% We keep the "greedy" approximate logic as a fallback for massive datasets,
% but strictly following the request, we implement the matching logic.

use_exact_matching = (n_a <= opts.max_exact_ambiguous);

if use_exact_matching
    
    if opts.verbose
        fprintf('Solving exact maximum weight matching for %d samples...\n', n_a);
    end

    % Construct slots
    % Slots map: slot_flat_idx -> (cluster_k)
    % To vectorize, we generate a list of cluster indices for all slots.
    
    % R_int is G x c. 
    % The matching is separable per group!
    % Constraint: sum_{i \in A_g} Z_{ik} = R_{gk} implies we only match samples in A_g to slots allocated for group g.
    % We can solve independent matching problems for each group g.
    
    current_label_update = zeros(n_a, 1);
    
    % Process each group separately
    for g = 1:G
        
        % Indices of ambiguous samples belonging to group g
        ambig_samples_in_g_indices_local = find(group(ambig_idx) == g); % index into ambig_idx
        ambig_samples_in_g_indices_global = ambig_idx(ambig_samples_in_g_indices_local);
        
        n_a_g = length(ambig_samples_in_g_indices_local);
        if n_a_g == 0
            continue;
        end
        
        % Slots for this group: R_int(g, :)
        quotas_g = R_int(g, :);
        
        % Check consistency (sum(quotas_g) should be n_a_g)
        if sum(quotas_g) ~= n_a_g
            % Numerical adjustment if off
             diff_g = n_a_g - sum(quotas_g);
             [~, max_q_idx] = max(quotas_g);
             quotas_g(max_q_idx) = max(0, quotas_g(max_q_idx) + diff_g);
        end
        
        % Expand slots for group g
        % slots_clusters(j) tells us which cluster slot j corresponds to
        slots_clusters = zeros(1, n_a_g);
        ptr = 1;
        for k = 1:c
            count = quotas_g(k);
            if count > 0
                slots_clusters(ptr : ptr + count - 1) = k;
                ptr = ptr + count;
            end
        end
        
        % Cost matrix for group g: [n_a_g x n_a_g] matches samples to slots
        % Cost(i, j) = -F(sample_i, cluster_of_slot_j)
        
        % Extract F for these samples: [n_a_g x c]
        F_sub_g = F(ambig_samples_in_g_indices_global, :);
        
        % Build cost matrix
        CostMat = zeros(n_a_g, n_a_g);
        
        % Depending on n_a_g size, vectorization:
        % CostMat(:, j) = -F_sub_g(:, slots_clusters(j))
        for j = 1:n_a_g
            CostMat(:, j) = -F_sub_g(:, slots_clusters(j));
        end
        
        % Solve LAP
        [assignment, ~] = matchpairs(CostMat, 1e12); % large cost for unassigned? no, standard call
        
        % Map back
        % assignment is [row, col] -> [sample_local_idx, slot_idx]
        for t = 1:size(assignment, 1)
             row_idx = assignment(t, 1);
             col_idx = assignment(t, 2);
             
             global_sample_idx = ambig_samples_in_g_indices_global(row_idx);
             assigned_cluster = slots_clusters(col_idx);
             
             label(global_sample_idx) = assigned_cluster;
        end
    end
    
else
    % Fallback: Greedy approximation for the matching problem (sort all possible edges by Weight)
    % This approximates Max Weight Matching.
    if opts.verbose
        fprintf('n_a=%d > max_exact (%d), using greedy approximation for matching.\n', n_a, opts.max_exact_ambiguous);
    end
    
    % For greedy, we also can process per group to simplify constraints.
    for g = 1:G
         ambig_samples_in_g_indices_local = find(group(ambig_idx) == g); 
         ambig_samples_in_g_indices_global = ambig_idx(ambig_samples_in_g_indices_local);
         
         n_a_g = length(ambig_samples_in_g_indices_local);
         if n_a_g == 0, continue; end
         
         quotas_g = R_int(g, :);
         
         % Local F
         F_sub_g = F(ambig_samples_in_g_indices_global, :);
         
         % Create list of all possible assignments (sample, cluster) with weights F
         % We need to carry quotas.
         % Heuristic: Just greedily assign highest F entries available.
         
         % Vectorize: (n_a_g * c) entries
         % [val, linear_idx] = sort(F_sub_g(:), 'descend');
         % [sample_idx, cluster_idx] = ind2sub(size(F_sub_g), linear_idx);
         
         % To avoid O(N*C) sort if C is large (usually C is small), this is fine.
         [vals, sort_indices] = sort(F_sub_g(:), 'descend');
         [rows, cols] = ind2sub(size(F_sub_g), sort_indices);
         
         assigned_mask = false(n_a_g, 1);
         curr_quotas = quotas_g;
         n_assigned = 0;
         
         for t = 1:length(vals)
             r = rows(t); % local sample index
             k = cols(t); % cluster index
             
             if ~assigned_mask(r) && curr_quotas(k) > 0
                 % Assign
                 assigned_mask(r) = true;
                 curr_quotas(k) = curr_quotas(k) - 1;
                 
                 global_idx = ambig_samples_in_g_indices_global(r);
                 label(global_idx) = k;
                 
                 n_assigned = n_assigned + 1;
                 if n_assigned == n_a_g
                     break;
                 end
             end
         end
         
         % Fallback for any unassigned (should not happen if sum quotas match n_a_g)
         if n_assigned < n_a_g
              unassigned_local = find(~assigned_mask);
              % Assign to any cluster with quota
              avail_clusters = [];
              for k=1:c
                  if curr_quotas(k) > 0
                      avail_clusters = [avail_clusters, repmat(k, 1, curr_quotas(k))];
                  end
              end
              
              % Just fill up
              for t = 1:length(unassigned_local)
                  if t <= length(avail_clusters)
                      global_idx = ambig_samples_in_g_indices_global(unassigned_local(t));
                      label(global_idx) = avail_clusters(t);
                  end
              end
         end
    end
end

end

function [res] = my_eval_y(y_pred, y_true, g_protected)
% my_eval_y: Evaluates clustering performance with 5 key metrics.
% ACC and NMI are computed using established library functions.
% NE, Bal, and f_CCE are implemented strictly following their definitions.
%
% Inputs:
%   y_pred:      Predicted cluster labels (n x 1 vector).
%   y_true:      Ground-truth cluster labels (n x 1 vector).
%   g_protected: Protected attribute labels (n x 1 vector).
%
% Outputs:
%   res: A column vector containing [ACC; NMI; NE; Bal; f_CCE].

% Ensure inputs are column vectors
y_pred = y_pred(:);
y_true = y_true(:);
g_protected = g_protected(:);

% --- 1 & 2. Standard Metrics: ACC and NMI ---
% These rely on an optimal mapping between predicted and true labels,
% handled by the external 'best_map' function.
y_pred_mapped = best_map(y_true, y_pred);

acc = mean(y_true == y_pred_mapped);
nmi = mutual_info(y_true, y_pred_mapped); % Assumes 'mutual_info' computes NMI


% --- 3. Balance Metric: NE (Normalized Entropy) ---
% Formula: NE = - (1/log(c)) * sum_{k=1 to c} (p_k * log(p_k))
% where p_k = |pi_k| / n
n = length(y_pred);
% Use the number of clusters from the prediction to handle empty clusters correctly
c_pred = length(unique(y_pred_mapped));
if c_pred <= 1
    ne = 1; % A single cluster is perfectly balanced in terms of entropy
else
    cluster_sizes = histcounts(y_pred_mapped, 1:c_pred+1)';
    p_k = cluster_sizes / n;
    
    % To handle p_k = 0, we compute the sum only over non-zero elements,
    % since lim_{p->0} p*log(p) = 0.
    non_zero_p_k = p_k(p_k > 0);
    entropy_val = -sum(non_zero_p_k .* log(non_zero_p_k));
    
    % Normalize by max entropy, log(c)
    ne = entropy_val / log(c_pred);
end


% --- 4. Fairness Metric: Bal (Balance) ---
% Formula: Bal = min_{k} ( N_k_min / N_k_max )
unique_pred_labels = unique(y_pred_mapped);
num_clusters_pred = length(unique_pred_labels);
unique_group_labels = unique(g_protected);
num_groups = length(unique_group_labels);

cluster_ratios = ones(num_clusters_pred, 1); % Default to 1 (perfectly balanced)

for k_idx = 1:num_clusters_pred
    k = unique_pred_labels(k_idx);
    
    % Get the protected group labels for samples in the current cluster k
    groups_in_cluster_k = g_protected(y_pred_mapped == k);
    
    if ~isempty(groups_in_cluster_k)
        % Count how many samples from each protected group are in this cluster
        group_counts_in_k = histcounts(groups_in_cluster_k, [unique_group_labels; unique_group_labels(end)+1])';
        
        N_k_min = min(group_counts_in_k);
        N_k_max = max(group_counts_in_k);
        
        if N_k_max > 0
            cluster_ratios(k_idx) = N_k_min / N_k_max;
        end
    end
end
bal = min(cluster_ratios);


% --- 5. Fairness Metric: f_CCE (fairness_Cluster Capacity Equality) ---
% Formula: f_CCE = min_{k} ( min_{i} ( min(c*gamma_ik, 1/(c*gamma_ik)) ) )
% where gamma_ik = |pi_k intersect G_i| / |G_i|
c = length(unique(y_true)); % Use ground-truth number of clusters for 'c' in formula

% Pre-calculate group sizes |G_i|
group_sizes = zeros(num_groups, 1);
for i = 1:num_groups
    group_sizes(i) = sum(g_protected == unique_group_labels(i));
end

s_ik_matrix = ones(num_groups, num_clusters_pred); % Default to 1

for k_idx = 1:num_clusters_pred
    k = unique_pred_labels(k_idx);
    cluster_k_mask = (y_pred_mapped == k);

    for i_idx = 1:num_groups
        i = unique_group_labels(i_idx);
        group_i_mask = (g_protected == i);
        
        intersection_size = sum(cluster_k_mask & group_i_mask);
        
        if group_sizes(i_idx) > 0
            gamma_ik = intersection_size / group_sizes(i_idx);
        else
            gamma_ik = 0;
        end
        
        % Calculate min(c*gamma, 1/(c*gamma))
        if gamma_ik == 0
            s_ik_matrix(i_idx, k_idx) = 0;
        else
            c_gamma = c * gamma_ik;
            s_ik_matrix(i_idx, k_idx) = min(c_gamma, 1/c_gamma);
        end
    end
end

f_cce_per_cluster = min(s_ik_matrix, [], 1); % Find min over all groups for each cluster
f_cce = min(f_cce_per_cluster);


% --- Consolidate Results ---
res = [acc; nmi; ne; bal; f_cce];
res = res(:); % Ensure output is a column vector

end
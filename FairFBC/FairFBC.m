function [label, U] = FairFBC(X, Yg, nCluster, param)

    %% 1. Pre-computation and Initialization
    [nSmp, ~] = size(X);
    [~, nGroup] = size(Yg);
    
    % --- Build the Fair Bipartite Graph and Pre-compute components ---
    % B is the global fair graph, A are the balanced anchors
    % Configure Graph Diffusion parameters
    graph_opts.anchor_replicates = 5;        % k-means replicates for anchor selection
    graph_opts.knn_graph_k = param.knn_size; % kNN graph connectivity
    graph_opts.diff_alpha = 0.15;           % PPR teleport probability
    graph_opts.diff_eps = 1e-4;             % APPR precision threshold
    graph_opts.diff_max_push = 2e6;         % Max push operations per anchor
    graph_opts.useParallel = false;          % Disable internal parallel processing
    graph_opts.seed_k = 1;                  % Seeds per anchor
    
    [B, A, anchor_ids] = buildFairBiGraph(X, Yg, param.m, param.knn_size, graph_opts);
    m = size(A, 1); % Actual number of anchors
    
    % Pre-compute Wg = Bg' * Bg for each group. This is a key optimization.
    Wg_list = cell(nGroup, 1);
    for g = 1:nGroup
        group_mask = (Yg(:, g) == 1);
        Bg = B(group_mask, :);
        Wg_list{g} = Bg' * Bg;
    end
    
    % --- High-Quality Initialization for U ---
    % Use PCA + k-means++ for better initialization, especially for high-dimensional data
    label_init_smp = smart_kmeans_init(X, nCluster);
    
    % Ensure labels are valid integers (handle NaN from k-means with missing data)
    invalid_mask = isnan(label_init_smp) | label_init_smp < 1 | label_init_smp > nCluster;
    if any(invalid_mask)
        label_init_smp(invalid_mask) = mod(find(invalid_mask) - 1, nCluster) + 1;
    end
    label_init_smp = round(label_init_smp);  % Ensure integer values
    
    % Convert sample labels to initial anchor labels U
    % Strategy: An anchor's initial label is the majority vote of the samples connected to it.
    F_init_hard = full(sparse(1:nSmp, label_init_smp, 1, nSmp, nCluster)); % n x c
    U_sum = B' * F_init_hard; % m x c, U_sum(i,j) = sum of connections from anchor i to samples in cluster j
    
    % Handle isolated anchors first, then normalize (correct order)
    row_sums = sum(U_sum, 2);  % Calculate row sums [m×1]
    zero_rows = find(row_sums < 1e-12);
    if ~isempty(zero_rows)
        % Assign very small values to isolated anchors before normalization
        U_sum(zero_rows, :) = 1e-8;  % Small uniform values
        row_sums = sum(U_sum, 2);    % Recalculate row sums
    end
    
    % Convert to probability distribution by normalizing rows
    U = U_sum ./ row_sums;  % Single normalization step
    

    
    %% 2. Alternating Optimization Loop
    obj_val = 0; % Initialize for convergence check


    iter = 1;
    
    while iter <= param.maxIter
        
        % In the new formulation, there is no W update step. W is effectively Identity.
        % We only perform Gradient Ascent on U followed by Simplex Projection.
        
        % --- Calculate Gradient w.r.t U ---
        % Gradient = sum_g Wg * U * (U' * Wg * U + epsilon * I)^(-1/2)
        grad_U = zeros(m, nCluster);
        
        for g = 1:nGroup
            Wg = Wg_list{g};
            
            % Compute M_g = U' * Wg * U
            Mg = U' * Wg * U;
            Mg_stable = stabilize_matrix(Mg, nCluster);
            
            % Compute (M_g + epsilon*I)^(-1/2)
            % Add epsilon for numerical stability as per paper
            epsilon_val = 1e-8; 
            Mg_eps = Mg_stable + epsilon_val * eye(nCluster);
            
            % Compute inverse square root
            Mg_inv_sqrt = inv(sqrtm(Mg_eps));
            
            % Accumulate gradient: Wg * U * Mg^(-1/2)
            grad_U = grad_U + Wg * U * Mg_inv_sqrt;
        end
        
        % --- Gradient Ascent Step ---
        % alpha (step size) search with backtracking
        alpha = 3 / (norm(grad_U, 'fro') + 1e-6);
        c = 1e-4; tau = 0.7; max_back = 15;
        if iter == 1
            base = compute_objective_value(U, Wg_list, nGroup, nCluster);
        else
             base = obj_val;
        end
        
        for t = 1:max_back
            U_temp = U + alpha * grad_U;
            U_try = project_to_simplex(U_temp);
            val = compute_objective_value(U_try, Wg_list, nGroup, nCluster);
            
            % Check sufficient increase
            % Note: Armijo condition typically uses gradient projection or just checking increase
            if val >= base + c * alpha * sum(grad_U(:).^2) 
                 U = U_try;
                 break;
            else
                 alpha = alpha * tau;
            end
             
            % Fallback if loop finishes without break
            if t == max_back
                U = U_try; % Take the last attempt
            end
        end
        
        % Update objective value from the accepted step
        obj_val = val; 
        
        % --- Check for Convergence ---
        if iter > 15
            if iter > 1
                 obj_change_ratio = abs(obj_val - prev_obj) / abs(prev_obj + 1e-10);
                 if obj_change_ratio < param.tolerance * 0.5
                     break;
                 end
            end
        end
        prev_obj = obj_val;
        
        iter = iter + 1;
    end
    
    %% 3. Finalization and Output
    
    % Truncate all objective histories to actual iterations
    if iter > param.maxIter
        iter = param.maxIter;
    end


    
    % Final assignment for data points is F = B * U
    F = B * U;
    
    % Fairness-aware hard assignment with optimized global assignment
    opts.ambig_quantile = 0.1;          
    opts.max_exact_ambiguous = 11000;    
    opts.verbose = false;               
    label = fairness_aware_assignment(F, Yg, nCluster, opts);
    
end

% --- Helper Function for Objective Value Computation ---
function obj_value = compute_objective_value(U, Wg_list, nGroup, nCluster)
% Computes value: sum_g Tr(sqrt(U' * Wg * U))
    obj_value = 0;
    for g = 1:nGroup
        Wg = Wg_list{g};
        Term_g = U' * Wg * U;
        Term_g_stable = stabilize_matrix(Term_g, nCluster);
        obj_value = obj_value + trace(sqrtm(Term_g_stable));
    end
    
    if ~isfinite(obj_value) || isnan(obj_value)
        obj_value = -inf;
    end
end

% --- Helper Function for Matrix Stabilization ---
function Term_g_stable = stabilize_matrix(Term_g, nCluster)
% Stabilizes a matrix using spectral radius adaptive perturbation
    Term_g_full = full(Term_g);
    Term_g_sym = (Term_g_full + Term_g_full') / 2;
    
    [V, D] = eig(Term_g_sym);
    d = diag(D); 
    d(d < 0) = 0; 
    
    eps0 = 1e-8 * (sum(d) / numel(d) + 1);
    d = d + eps0;
    
    Term_g_stable = V * diag(d) * V';
end

% --- Helper Function for Simplex Projection ---
function P = project_to_simplex(V)
% Projects each row of V onto the probability simplex.
    [m, c] = size(V);
    P = zeros(m, c);
    for i = 1:m
        v_row = V(i, :);
        v_sorted = sort(v_row, 'descend');
        cumsum_v = cumsum(v_sorted);
        rho = find(v_sorted > (cumsum_v - 1) ./ (1:c), 1, 'last');
        if isempty(rho)
            rho = c;
        end
        theta = (cumsum_v(rho) - 1) / rho;
        P(i, :) = max(v_row - theta, 0);
    end
end

% --- Helper Function for Smart K-means Initialization ---
function [label, center] = smart_kmeans_init(X, k)
    [n, d] = size(X);
    nan_mask = isnan(X);
    if any(nan_mask(:))
        for j = 1:d
            col_mean = mean(X(~nan_mask(:,j), j));
            if ~isnan(col_mean)
                X(nan_mask(:,j), j) = col_mean;
            else
                X(nan_mask(:,j), j) = 0;
            end
        end
    end

    if d >= 512
        npc = 128;
        [coeff, score] = pca(X, 'NumComponents', npc);
        opts = statset('MaxIter', 100, 'Display', 'off');
        [label, ~] = kmeans(score, k, ...
            'Start', 'plus', 'Replicates', 5, ...
            'Options', opts);
        center = zeros(k, d);
        for i = 1:k
            center(i, :) = mean(X(label == i, :), 1);
        end
    else
        opts = statset('MaxIter', 100, 'Display', 'off');
        [label, center] = kmeans(X, k, ...
            'Start', 'plus', 'Replicates', 5, ...
            'Options', opts);
    end
end

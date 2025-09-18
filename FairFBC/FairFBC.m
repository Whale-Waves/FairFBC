function [label, U, W, objHistory, objHistory_afterW, objHistory_afterU] = FairFBC(X, Yg, nCluster, param)

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
    graph_opts.useParallel = true;          % Enable parallel processing
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
    
    % --- Advanced Initialization for W based on Anchor Centrality ---
    anchor_degrees = sum(B, 1)'; % [m×1] vector of anchor degrees
    w_diag_init = anchor_degrees ./ (sum(anchor_degrees) + 1e-10); % Normalize to probability simplex: sum = 1
    W = spdiags(w_diag_init, 0, m, m); % Initialize W with centrality-based weights
    
    %% 2. Alternating Optimization Loop
    objHistory = zeros(param.maxIter, 1);  % Pre-allocate for efficiency (overall objective)
    objHistory_afterW = zeros(param.maxIter, 1);  % Objective after W update
    objHistory_afterU = zeros(param.maxIter, 1);  % Objective after U update
    iter = 1;
    
    while iter <= param.maxIter
        
        % --- Step A: Fix U, Update w (Principled Optimization) ---
        
        % Extract current weight vector from diagonal matrix W
        w_diag = full(diag(W));  % Convert current W to weight vector
        
        % 1. Calculate gradient of objective function w.r.t. each w_i
        grad_w = zeros(m, 1);
        
        for g = 1:nGroup
            Wg = Wg_list{g};
            
            % Current objective term: X_g = U' * diag(w) * Wg * U
            Term_g = U' * W * Wg * U;  % W is diag(w_old) from previous iteration
            Term_g_stable = stabilize_matrix(Term_g, nCluster);
            
            % Compute (1/2) * X_g^(-1/2) for gradient calculation
            Term_g_sqrt = sqrtm(Term_g_stable + 1e-8 * eye(nCluster));
            Term_g_inv_sqrt = 0.5 * inv(Term_g_sqrt);
            
            % Efficient gradient calculation using matrix operations
            grad_matrix = Wg * U * Term_g_inv_sqrt * U';
            grad_w = grad_w + diag(grad_matrix);
        end
        
        % 2. Mirror descent (exponential gradient) for w - more stable and faster
        eta = 2 / (norm(grad_w, 2) + 1e-9);
        w_candidate = w_diag .* exp(eta * grad_w);    % Mirror step - naturally non-negative
        w_diag = w_candidate / (sum(w_candidate) + 1e-12);  % Normalize to simplex
        
        % 3. Construct the new weight matrix W
        W = spdiags(w_diag, 0, m, m);
        
        % --- Calculate Objective after W Update ---
        obj_after_W = compute_objective_value(U, W, Wg_list, nGroup, nCluster);
        objHistory_afterW(iter) = obj_after_W;
        
        % --- Step B: Fix W, Update U (using Projected Gradient Ascent) ---
        % Calculate the gradient of the objective w.r.t. U
        grad_U = zeros(m, nCluster);
        for g = 1:nGroup
            Wg = Wg_list{g};
            Term_g = U' * W * Wg * U;
            Term_g_stable = stabilize_matrix(Term_g, nCluster);
            Term_g_sqrt = sqrtm(Term_g_stable + 1e-8 * eye(nCluster));
            grad_U = grad_U + (W * Wg * U) / Term_g_sqrt;  
        end
        
        % Gradient Ascent Step with backtracking line search (Armijo condition)
        alpha = 3 / (norm(grad_U, 'fro') + 1e-6);
        c = 1e-4; tau = 0.7; max_back = 15;
        base = compute_objective_value(U, W, Wg_list, nGroup, nCluster);
        
        for t = 1:max_back
            U_try = project_to_simplex(U + alpha * grad_U);
            val = compute_objective_value(U_try, W, Wg_list, nGroup, nCluster);
            if val >= base + c * alpha * sum(grad_U(:).^2)  % Armijo condition
                U = U_try; 
                break;
            else
                alpha = alpha * tau;  % Backtrack
            end
        end
        
        % --- Calculate and Record Objective Value after U Update ---
        obj_after_U = compute_objective_value(U, W, Wg_list, nGroup, nCluster);
        objHistory_afterU(iter) = obj_after_U;
        objHistory(iter) = obj_after_U;  % Keep overall history for compatibility
        
        % --- Check for Convergence ---
        if iter > 15
            obj_change_ratio = abs(objHistory(iter) - objHistory(iter-1)) / abs(objHistory(iter-1) + 1e-10);
            if obj_change_ratio < param.tolerance * 0.5
                break;
            end
        end
        
        iter = iter + 1;
    end
    
    %% 3. Finalization and Output
    
    % Truncate all objective histories to actual iterations
    % Handle both convergence break and normal loop completion cases
    if iter > param.maxIter
        iter = param.maxIter;
    end
    objHistory = objHistory(1:iter);
    objHistory_afterW = objHistory_afterW(1:iter);
    objHistory_afterU = objHistory_afterU(1:iter);
    
    % Final assignment for data points is F = B * U
    F = B * U;
    
    % Fairness-aware hard assignment with optimized global assignment
    % Configure parameters for optimal performance
    opts.ambig_quantile = 0.1;          
    opts.max_exact_ambiguous = 11000;    
    opts.verbose = false;               
    label = fairness_aware_assignment(F, Yg, nCluster, opts);
    
end

% --- Helper Function for Objective Value Computation ---
function obj_value = compute_objective_value(U, W, Wg_list, nGroup, nCluster)
% Computes the unified objective function value: sum_g Tr(sqrt(U' * W * Wg * U))
    obj_value = 0;
    for g = 1:nGroup
        Wg = Wg_list{g};
        Term_g = U' * W * Wg * U;
        Term_g_stable = stabilize_matrix(Term_g, nCluster);
        obj_value = obj_value + trace(sqrtm(Term_g_stable));
    end
    
    % Handle numerical issues
    if ~isfinite(obj_value) || isnan(obj_value)
        obj_value = -inf;  % Return a penalty value for invalid results
    end
end

% --- Helper Function for Matrix Stabilization ---
function Term_g_stable = stabilize_matrix(Term_g, nCluster)
% Stabilizes a matrix using spectral radius adaptive perturbation (faster and more stable)
    % Convert to full matrix to avoid sparse matrix issues
    Term_g_full = full(Term_g);
    
    % Ensure matrix is symmetric (for reliable eigenvalue computation)
    Term_g_sym = (Term_g_full + Term_g_full') / 2;
    
    % Eigenvalue decomposition for PSD enforcement
    [V, D] = eig(Term_g_sym);
    d = diag(D); 
    d(d < 0) = 0;  % PSD clipping - remove negative eigenvalues
    
    % Adaptive perturbation based on spectral scale
    eps0 = 1e-8 * (sum(d) / numel(d) + 1);  % Scale with average eigenvalue
    d = d + eps0;  % Add small perturbation to all eigenvalues
    
    % Reconstruct stabilized matrix
    Term_g_stable = V * diag(d) * V';
end

% --- Helper Function for Simplex Projection ---
function P = project_to_simplex(V)
% Projects each row of V onto the probability simplex.
    [m, c] = size(V);
    P = zeros(m, c);
    for i = 1:m
        v_row = V(i, :);
        % A fast projection algorithm
        v_sorted = sort(v_row, 'descend');
        cumsum_v = cumsum(v_sorted);
        rho = find(v_sorted > (cumsum_v - 1) ./ (1:c), 1, 'last');
        if isempty(rho)  % Handle edge case
            rho = c;
        end
        theta = (cumsum_v(rho) - 1) / rho;
        P(i, :) = max(v_row - theta, 0);
    end
end

% --- Helper Function for Smart K-means Initialization ---
function [label, center] = smart_kmeans_init(X, k)

    [n, d] = size(X);
    
    % Handle NaN values by mean imputation (simple but effective)
    nan_mask = isnan(X);
    if any(nan_mask(:))
        for j = 1:d
            col_mean = mean(X(~nan_mask(:,j), j));
            if ~isnan(col_mean)
                X(nan_mask(:,j), j) = col_mean;
            else
                X(nan_mask(:,j), j) = 0;  % fallback if entire column is NaN
            end
        end
    end

    if d >= 512
        npc = 128;  % reduce dimension
        [coeff, score] = pca(X, 'NumComponents', npc);
        % kmeans++ on reduced features
        opts = statset('MaxIter', 100, 'Display', 'off');
        [label, ~] = kmeans(score, k, ...
            'Start', 'plus', 'Replicates', 5, ...
            'Options', opts);
        % map back to original space
        center = zeros(k, d);
        for i = 1:k
            center(i, :) = mean(X(label == i, :), 1);
        end
    else
        % Direct kmeans++ without PCA
        opts = statset('MaxIter', 100, 'Display', 'off');
        [label, center] = kmeans(X, k, ...
            'Start', 'plus', 'Replicates', 5, ...
            'Options', opts);
    end
end


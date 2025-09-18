%% 1. Environment Setup
clear;
clc;
close all;

% --- Add necessary paths ---
script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'data'));
addpath(fullfile(script_directory, 'lib'));
fprintf('Paths added successfully.\n\n');

%% 2. Experiment Configuration
exp_name = 'FairFBC_ablation_anchors1'; % A name for this experiment run with fmincon-optimized step size

% --- Parameter Selection Strategy ---
% 'acc_fcce' 
% 'acc_nmi'  
pareto_strategy = 'acc_fcce';  

% --- Find all datasets in the 'data' folder ---
data_dir_info = dir(fullfile(script_directory, 'data', '*.mat'));
all_datasets = {data_dir_info.name};

% --- Select which datasets to run  ---
% Option 1: Manually specify datasets to run
%datasets_to_run = {'adult_race_32561n_107d_2c_5g.mat', 'german_foreign_1000n_24d_2c_2g.mat'};

% Option 2: Run all found datasets
datasets_to_run = all_datasets;

fprintf('============================================================\n');
fprintf('Starting Experiment: %s\n', exp_name);
fprintf('Found %d datasets. Will process %d of them.\n', length(all_datasets), length(datasets_to_run));
fprintf('============================================================\n\n');

%% 3. Main Experiment Loop
for i_run = 1:length(datasets_to_run)
    data_name_with_ext = datasets_to_run{i_run};
    data_name = strrep(data_name_with_ext, '.mat', '');

    fprintf('##### Processing Dataset [%d/%d]: %s #####\n', i_run, length(datasets_to_run), data_name);
    
    % --- Create directory for results ---
    results_dir = fullfile(script_directory, 'results', exp_name, data_name);
    if ~exist(results_dir, 'dir'), mkdir(results_dir); end
    
    % --- Check if results already exist to avoid re-computation ---
    fname_result = fullfile(results_dir, [data_name, '_result.mat']);
    if exist(fname_result, 'file')
        fprintf('  -> Results for %s already exist. Skipping.\n\n', data_name);
        continue;
    end
    
    % --- Load Data ---
    fprintf('  -> Loading data (x, y, g)...\n');
    load(data_name_with_ext, 'x', 'y', 'g');
    
    % --- Data Type Conversion ---
    % Convert to double precision to avoid sparse matrix compatibility issues
    if ~isa(x, 'double')
        fprintf('  -> Converting data from %s to double precision...\n', class(x));
        x = double(x);
    end
    
    % --- Data Preparation ---
    [nSmp, ~] = size(x);
    nCluster = length(unique(y));
    
    % Robust conversion of group labels to one-hot matrix
    g = g(:); % Ensure g is a column vector
    unique_groups = unique(g);
    nGroup = length(unique_groups);
    Yg = zeros(nSmp, nGroup);
    for i = 1:nGroup
        Yg(:, i) = (g == unique_groups(i));
    end
    
    fprintf('  -> Data loaded: %d samples, %d clusters, %d sensitive groups.\n', nSmp, nCluster, nGroup);
    
    % --- Parameter Configuration for Grid Search ---
    
    base_m_multipliers = [];
    base_m_values = nCluster * base_m_multipliers;
    additional_m_values = [4, 16, 24, 64, 512,5120];
    candidate_m_range = [base_m_values, additional_m_values];
    
    
    m_range = sort(unique(candidate_m_range(candidate_m_range >= nGroup)));
    knn_range = [5, 10, 15, 20];           
    n_repeats = 10;                       
    
    fprintf('  -> Valid m range after group constraint: [%s]\n', num2str(m_range));
    fprintf('  -> Minimum anchors required (one per group): %d\n', nGroup);
    
    % Fixed parameters
    maxIter = 50;             
    tolerance = 1e-5;            
    
    fprintf('  -> Grid search setup: %d m values, %d knn values, %d repeats each\n', ...
            length(m_range), length(knn_range), n_repeats);
    
    % --- Random Seed Setup for Reproducible Experiments ---
    rng(2026, 'twister');
    random_seeds = randi(1e6, 1, n_repeats);
    
    % --- Execute Grid Search ---
    fprintf('  -> Starting grid search for optimal parameters...\n');
    
    % Initialize storage for all results (pre-allocation for performance)
    total_results = length(m_range) * length(knn_range) * n_repeats;
    all_results = struct('m', cell(total_results, 1), 'knn_size', [], 'repeat', [], ...
                        'metrics', [], 'ACC', [], 'NMI', [], 'NE', [], 'Bal', [], 'f_CCE', [], ...
                        'predicted_labels', [], 'final_U', [], 'final_W', [], ...
                        'objective_history', [], 'runtime_seconds', [], 'parameters', []);
    result_idx = 1;
    
    % Ensure labels are in the correct column vector format (do this once)
    y = y(:);
    % Note: g already converted to column vector at line 78
    
    % Grid search over parameter combinations (knn outer, m inner)
    param_combo_idx = 0;
    total_combos = length(knn_range) * length(m_range);
    
    for i_knn = 1:length(knn_range)
        for i_m = 1:length(m_range)
            param_combo_idx = param_combo_idx + 1;
            knn_val = knn_range(i_knn);
            m_val = m_range(i_m);
            
            % Storage for this parameter combination (pre-allocated)
            combo_acc = zeros(n_repeats, 1);
            combo_nmi = zeros(n_repeats, 1);
            combo_f_cce = zeros(n_repeats, 1);
            combo_time = zeros(n_repeats, 1);
            
            % Multiple runs for this parameter combination (silent)
            for i_repeat = 1:n_repeats
                % Set random seed for this repeat
                rng(random_seeds(i_repeat), 'twister');
                
                % Set parameters for this run
                param = struct();
                param.m = m_val;
                param.knn_size = knn_val;
                param.maxIter = maxIter;
                param.tolerance = tolerance;
                
                % Run algorithm
                tic;
                [label, U_final, W_final, objHistory, objHistory_afterW, objHistory_afterU] = FairFBC(x, Yg, nCluster, param);
                run_time = toc;
                
                % Evaluate performance
                label = label(:);
                eval_metrics = my_eval_y(label, y, g);
                
                % Store for averaging
                combo_acc(i_repeat) = eval_metrics(1);
                combo_nmi(i_repeat) = eval_metrics(2);
                combo_f_cce(i_repeat) = eval_metrics(5);
                combo_time(i_repeat) = run_time;
                
                % Store detailed results
                all_results(result_idx).m = m_val;
                all_results(result_idx).knn_size = knn_val;
                all_results(result_idx).repeat = i_repeat;
                all_results(result_idx).metrics = eval_metrics;
                all_results(result_idx).ACC = eval_metrics(1);
                all_results(result_idx).NMI = eval_metrics(2);
                all_results(result_idx).NE = eval_metrics(3);
                all_results(result_idx).Bal = eval_metrics(4);
                all_results(result_idx).f_CCE = eval_metrics(5);
                all_results(result_idx).predicted_labels = label;
                all_results(result_idx).final_U = U_final;
                all_results(result_idx).final_W = W_final;
                all_results(result_idx).objective_history = objHistory;
                all_results(result_idx).objective_history_afterW = objHistory_afterW;
                all_results(result_idx).objective_history_afterU = objHistory_afterU;
                all_results(result_idx).runtime_seconds = run_time;
                all_results(result_idx).parameters = param;
                
                result_idx = result_idx + 1;
            end
            
            % Display averaged results for this parameter combination
            avg_acc = mean(combo_acc);
            avg_nmi = mean(combo_nmi);
            avg_f_cce = mean(combo_f_cce);
            total_time = sum(combo_time);  % Total time for all repeats
            fprintf('    [%d/%d] knn=%d, m=%d: ACC=%.3f±%.3f, NMI=%.3f±%.3f, f_CCE=%.3f±%.3f (%.1fs)\n', ...
                    param_combo_idx, total_combos, knn_val, m_val, ...
                    avg_acc, std(combo_acc), avg_nmi, std(combo_nmi), avg_f_cce, std(combo_f_cce), total_time);
        end
    end
    
    % --- Find Pareto Optimal Parameters ---
    fprintf('  -> Finding Pareto optimal parameters (%s strategy)...\n', pareto_strategy);
    
    % Compute average metrics for each parameter combination (pre-allocated)
    total_param_combos = length(knn_range) * length(m_range);
    param_combinations = struct('m', cell(total_param_combos, 1), 'knn_size', [], ...
                               'ACC', [], 'NMI', [], 'NE', [], 'Bal', [], 'f_CCE', []);
    param_idx = 1;
    
    for i_knn = 1:length(knn_range)
        for i_m = 1:length(m_range)
            knn_val = knn_range(i_knn);
            m_val = m_range(i_m);
            
            % Find all results for this combination
            mask = ([all_results.m] == m_val) & ([all_results.knn_size] == knn_val);
            combo_results = all_results(mask);
            
            % Compute averages for all 5 metrics
            param_combinations(param_idx).m = m_val;
            param_combinations(param_idx).knn_size = knn_val;
            param_combinations(param_idx).ACC = mean([combo_results.ACC]);
            param_combinations(param_idx).NMI = mean([combo_results.NMI]);
            param_combinations(param_idx).NE = mean([combo_results.NE]);
            param_combinations(param_idx).Bal = mean([combo_results.Bal]);
            param_combinations(param_idx).f_CCE = mean([combo_results.f_CCE]);
            
            param_idx = param_idx + 1;
        end
    end
    
    % Apply selected Pareto optimization strategy
    switch pareto_strategy
        case 'acc_fcce'
            best_params = findBestParamsAccFCCE(param_combinations);
        case 'acc_nmi'
            best_params = findBestParamsAccNMI(param_combinations);
        otherwise
            error('Unknown pareto_strategy: %s. Use ''acc_fcce'' or ''acc_nmi''.', pareto_strategy);
    end
    
    % --- Calculate average runtime for best parameters ---
    best_m = best_params.m;
    best_knn = best_params.knn_size;
    
    % Find all results for best parameter combination
    best_mask = ([all_results.m] == best_m) & ([all_results.knn_size] == best_knn);
    best_combo_results = all_results(best_mask);
    best_avg_runtime = mean([best_combo_results.runtime_seconds]);
    
    % --- Save Results ---
    fprintf('  -> Saving grid search results to file...\n');
    
    results.best_parameters = best_params;
    results.best_parameters.avg_runtime = best_avg_runtime;  % Add average runtime for best params
    results.param_results = param_combinations;
    results.all_results = all_results;  % Save detailed results for visualization
    
    save(fname_result, 'results');
    
    fprintf('  -> Results saved successfully to %s\n', fname_result);
    fprintf('  -> Best parameters runtime: %.3f seconds/run\n', best_avg_runtime);
    fprintf('##### Finished processing %s #####\n\n', data_name);

end

fprintf('============================================================\n');
fprintf('All datasets have been processed. Experiment finished.\n');
fprintf('============================================================\n');

%% 6. Cleanup
rmpath(fullfile(script_directory, 'data'));
rmpath(fullfile(script_directory, 'lib'));
fprintf('Paths removed.\n');
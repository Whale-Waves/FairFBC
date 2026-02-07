%% 1. Environment Setup
clear;
clc;
close all;

% --- Add necessary paths ---
script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'data'));
addpath(fullfile(script_directory, 'lib'));
fprintf('Paths added successfully.\n\n');

% --- Parallel pool setup removed ---


%% 2. Experiment Configuration
exp_name = 'FairFBC'; 
pareto_strategy = 'acc_fcce';  

% --- Find all datasets in the 'data' folder ---
data_dir_info = dir(fullfile(script_directory, 'data', '*.mat'));
all_datasets = {data_dir_info.name};

% --- Select which datasets to run  ---
datasets_to_run = {'fairseg_race_10000n_512d_2c_3g.mat'};

fprintf('============================================================\n');
fprintf('Starting Experiment: %s\n', exp_name);
fprintf('Found %d datasets. Will process %d of them.\n', length(all_datasets), length(datasets_to_run));
fprintf('============================================================\n\n');

%% 3. Main Experiment Loop (Process datasets sequentially)
for i_run = 1:length(datasets_to_run)
    data_name_with_ext = datasets_to_run{i_run};
    data_name = strrep(data_name_with_ext, '.mat', '');

    fprintf('##### Processing Dataset [%d/%d]: %s #####\n', i_run, length(datasets_to_run), data_name);
    
    % --- Create directory for results ---
    results_dir = fullfile(script_directory, 'results', exp_name, data_name);
    if ~exist(results_dir, 'dir'), mkdir(results_dir); end
    
    % --- Check if results already exist ---
    fname_result = fullfile(results_dir, [data_name, '_result.mat']);
    if exist(fname_result, 'file')
        fprintf('  -> Results for %s already exist. Skipping.\n\n', data_name);
        continue;
    end
    
    % --- Load Data ---
    fprintf('  -> Loading data (x, y, g)...\n');
    load(data_name_with_ext, 'x', 'y', 'g');
    
    if ~isa(x, 'double')
        x = double(x);
    end
    
    [nSmp, ~] = size(x);
    nCluster = length(unique(y));
    
    g = g(:); 
    unique_groups = unique(g);
    nGroup = length(unique_groups);
    Yg = zeros(nSmp, nGroup);
    for i = 1:nGroup
        Yg(:, i) = (g == unique_groups(i));
    end
    y = y(:);
    
    fprintf('  -> Data loaded: %d samples, %d clusters, %d sensitive groups.\n', nSmp, nCluster, nGroup);
    
    % --- Parameter Configuration ---
    base_m_multipliers = [];
    base_m_values = nCluster * base_m_multipliers;
    additional_m_values = [4, 16, 24, 64, 512];
    candidate_m_range = [base_m_values, additional_m_values];
    
    m_range = sort(unique(candidate_m_range(candidate_m_range >= nGroup)));
    knn_range = [5, 10, 15, 20];           
    n_repeats = 1;                       
    
    % Fixed parameters
    maxIter = 50;             
    tolerance = 1e-5;            
    
    % --- Prepare Parameter List for Parallel Execution ---
    % Flatten the loops into a single list of tasks
    tasks = [];
    task_idx = 0;
    for i_knn = 1:length(knn_range)
        for i_m = 1:length(m_range)
            for i_rep = 1:n_repeats
                task_idx = task_idx + 1;
                t = struct();
                t.m = m_range(i_m);
                t.knn = knn_range(i_knn);
                t.rep = i_rep;
                t.id = task_idx;
                if isempty(tasks)
                    tasks = t;
                else
                    tasks(end+1) = t;
                end
            end
        end
    end
    
    num_tasks = length(tasks);
    fprintf('  -> Starting SERIAL grid search: %d total tasks...\n', num_tasks);
    
    % Initialize results container for PARFOR
    % Note: Cannot index struct array directly in random order in parfor
    % We will save each task result to a cell array
    par_results_cell = cell(num_tasks, 1);
    
    % --- SERIAL LOOP ---
    for t_idx = 1:num_tasks
        task = tasks(t_idx);
        
         % Set random seed 
        % In parfor, rng needs care. Combining task parameters ensures unique stream.
        rng(2026 + task.id * 100, 'twister'); 
        
        % Parameters
        param = struct();
        param.m = task.m;
        param.knn_size = task.knn;
        param.maxIter = maxIter;
        param.tolerance = tolerance;
        
        % Run Algorithm
        t_start = tic;
        % FairFBC must NOT use internal parallelization
        % FairFBC serialized call
        [label, U_final] = FairFBC(x, Yg, nCluster, param);
        run_time = toc(t_start);
        
        % Evaluate
        label = label(:);
        eval_metrics = my_eval_y(label, y, g);
        
        % Store Result
        res = struct();
        res.m = task.m;
        res.knn_size = task.knn;
        res.repeat = task.rep;
        res.metrics = eval_metrics;
        res.ACC = eval_metrics(1);
        res.NMI = eval_metrics(2);
        res.NE = eval_metrics(3);
        res.Bal = eval_metrics(4);
        res.f_CCE = eval_metrics(5);
        res.predicted_labels = label;
        res.final_U = U_final;
        % res.final_W = W_final; % Removed
        % res.objective_history = objHistory; % Removed
        % res.objective_history_afterW = objHistory_afterW; % Removed
        % res.objective_history_afterU = objHistory_afterU; % Removed
        res.runtime_seconds = run_time;
        res.parameters = param;
        
        par_results_cell{t_idx} = res;
        
        % Minimal printing inside parfor
        fprintf('    [Task %3d/%3d] knn=%d, m=%d: ACC=%.3f, f_CCE=%.3f (%.1fs)\n', ...
                t_idx, num_tasks, task.knn, task.m, res.ACC, res.f_CCE, run_time);
    end
    
    % --- Reassemble Results ---
    % Convert cell array back to struct array
    all_results = [par_results_cell{:}];
    
    % --- Find Pareto (Post-processing) ---
    fprintf('  -> Finding Pareto optimal parameters (Post-processing)...\n');
    
    % Re-calculate averages (since we flattened loops)
    param_combinations = [];
    p_idx = 0;
    
    for i_knn = 1:length(knn_range)
        for i_m = 1:length(m_range)
            knn_v = knn_range(i_knn);
            m_v = m_range(i_m);
            
            % Filter results
            mask = ([all_results.m] == m_v) & ([all_results.knn_size] == knn_v);
            combo_res = all_results(mask);
            
            p_idx = p_idx + 1;
            pc = struct();
            pc.m = m_v;
            pc.knn_size = knn_v;
            pc.ACC = mean([combo_res.ACC]);
            pc.NMI = mean([combo_res.NMI]);
            pc.NE = mean([combo_res.NE]);
            pc.Bal = mean([combo_res.Bal]);
            pc.f_CCE = mean([combo_res.f_CCE]);
            
            if isempty(param_combinations)
                param_combinations = pc;
            else
                param_combinations(p_idx) = pc;
            end
        end
    end
    
    switch pareto_strategy
        case 'acc_fcce'
            best_params = findBestParamsAccFCCE(param_combinations);
        case 'acc_nmi'
            best_params = findBestParamsAccNMI(param_combinations);
        otherwise
            best_params = findBestParamsAccFCCE(param_combinations);
    end
    
    % --- Save Results ---
    % Find runtime for best params
    best_mask = ([all_results.m] == best_params.m) & ([all_results.knn_size] == best_params.knn_size);
    best_avg_runtime = mean([all_results(best_mask).runtime_seconds]);
    
    results.best_parameters = best_params;
    results.best_parameters.avg_runtime = best_avg_runtime;
    results.param_results = param_combinations;
    results.all_results = all_results;
    
    save(fname_result, 'results');
    fprintf('  -> Results saved to %s\n', fname_result);
    fprintf('##### Finished processing %s #####\n\n', data_name);

end

fprintf('============================================================\n');
fprintf('Experiment finished.\n');
fprintf('============================================================\n');

%% Cleanup
rmpath(fullfile(script_directory, 'data'));
rmpath(fullfile(script_directory, 'lib'));
% delete(gcp('nocreate')); % Removed

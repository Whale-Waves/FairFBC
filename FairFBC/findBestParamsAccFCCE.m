function best_params = findBestParamsAccFCCE(param_combinations)

    % Extract ACC and f_CCE values
    acc_vals = [param_combinations.ACC];
    f_cce_vals = [param_combinations.f_CCE];
    
    % Compute distance to ideal point [1,1] in ACC-f_CCE space
    distances = sqrt((acc_vals - 1).^2 + (f_cce_vals - 1).^2);
    
    % Find parameter combination with minimum distance to ideal point
    [~, best_idx] = min(distances);
    best_params = param_combinations(best_idx);
    
    fprintf('  -> Pareto selection (ACC vs f_CCE): m=%d, knn_size=%d (ACC=%.4f, f_CCE=%.4f)\n', ...
            best_params.m, best_params.knn_size, best_params.ACC, best_params.f_CCE);
end

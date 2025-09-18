function best_params = findBestParamsAccNMI(param_combinations)

    % Extract ACC and NMI values
    acc_vals = [param_combinations.ACC];
    nmi_vals = [param_combinations.NMI];
    
    % Compute distance to ideal point [1,1] in ACC-NMI space
    distances = sqrt((acc_vals - 1).^2 + (nmi_vals - 1).^2);
    
    % Find parameter combination with minimum distance to ideal point
    [~, best_idx] = min(distances);
    best_params = param_combinations(best_idx);
    
    fprintf('  -> Pareto selection (ACC vs NMI): m=%d, knn_size=%d (ACC=%.4f, NMI=%.4f, f_CCE=%.4f)\n', ...
            best_params.m, best_params.knn_size, best_params.ACC, best_params.NMI, best_params.f_CCE);
end

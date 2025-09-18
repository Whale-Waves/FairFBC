clear;
clc;
addpath(pwd);
data_result_dir = fullfile(pwd, 'data_result');
output_root = fullfile(pwd, 'metrics_result');
if ~exist(output_root, 'dir'), mkdir(output_root); end

reference_dir = fullfile(data_result_dir, 'FairFBC');
if ~exist(reference_dir, 'dir')
    error('基准目录不存在: %s', reference_dir);
end
ref_files = dir(fullfile(reference_dir, '*_result.mat'));
dataset_names = {};
for i = 1:length(ref_files)
    filename = ref_files(i).name;
    full_name = strrep(filename, '_result.mat', '');
    underscore_pos = strfind(full_name, '_');
    if ~isempty(underscore_pos)
        dataset_name = full_name(1:underscore_pos(1)-1);
    else
        dataset_name = full_name;
    end
    if ~ismember(dataset_name, dataset_names)
        dataset_names{end+1} = dataset_name;
    end
end

fprintf('发现 %d 个数据集：\n', numel(dataset_names));
for i = 1:numel(dataset_names), fprintf('  %d) %s\n', i, dataset_names{i}); end

algorithm_config = {
    {'F3KM',       'F3KM_Run',     'results.param_results',   '',     'struct'}
    {'FairFBC',    'FairFBC',      'results.param_results',   '',     'struct'}
    {'FairFBC-f',  'FairFBC-f',    'results.param_results',   '',     'struct'}
    {'FairGNN',    'FairGNN_Run',  'param_results',           '',     'cell'  }
    {'FCE',        'FCE_result',   'results.param_results',   'avg_', 'struct'}
    {'FFC',        'FFC_Run',      'results.param_results',   '',     'cell'  }
    {'FKCC',       'FKCC_result',  'results.param_results',   '',     'struct'}
    {'FKKM',       'FKKM_result',  'results.param_results',   '',     'struct'}
    {'FSPGA',      'FSPGA_result', 'results.param_results',   '',     'struct'}
    {'iFairNMTF',  'iFairNMTF_Run','results.param_results',   '',     'cell'  }
    {'KFC',        'KFC_Run',      'results.param_results',   '',     'cell'  }
    {'VFC',        'VFC_Run',      'results.param_results',   '',     'cell'  }
};

ref_point = [0,0];
zstar     = [1,1];
W         = generate_weights_2d(200);
tau_grid  = linspace(0,1,21);

all_results = {};
result_idx = 0;

for d = 1:numel(dataset_names)
    ds = dataset_names{d};
    methods = []; midx = 0;

    % 读取各算法数据
    for i = 1:size(algorithm_config,1)
        cfg = algorithm_config{i};
        name   = cfg{1}; folder = cfg{2};
        path_s = cfg{3}; prefix = cfg{4}; type_s = cfg{5};

        pattern_files = dir(fullfile(data_result_dir, folder, [ds '_*.mat']));
        if isempty(pattern_files)
            fprintf('警告：未找到匹配 %s 的文件（跳过 %s）\n', ds, name);
            continue;
        end
        fpath = fullfile(data_result_dir, folder, pattern_files(1).name);
        S = load(fpath);
        try
            P = extract_acc_fcce(S, path_s, type_s, prefix); % -> [ACC, f_CCE]
            if isempty(P)
                fprintf('警告：%s/%s 未提取到有效(ACC, f_CCE)，跳过\n', folder, ds);
                continue;
            end
            midx = midx + 1;
            methods(midx).name = name; %#ok<SAGROW>
            methods(midx).P    = P;    %#ok<SAGROW>
        catch ME
            fprintf('警告：%s 解析失败（%s）\n', name, ME.message);
        end
    end

    if isempty(methods)
        fprintf('数据集 %s：未找到任何方法的有效结果，跳过。\n', ds);
        continue;
    end

    % 参考前沿 R：用所有方法并集的非支配点
    allP = [];
    for j = 1:numel(methods), allP = [allP; methods(j).P]; end %#ok<AGROW>
    R = pareto_front_max(allP);

    % 计算指标并排序
    T = compute_metrics_for_methods(methods, ref_point, zstar, W, R);
    
    % 收集结果数据
    for j = 1:numel(methods)
        result_idx = result_idx + 1;
        all_results{result_idx} = struct(...
            'Dataset', ds, ...
            'Algorithm', methods(j).name, ...
            'HV', T.HV(j), ...
            'IGDplus', T.IGDplus(j), ...
            'R2', T.R2(j), ...
            'NumPoints', size(methods(j).P, 1) ...
        );
    end
    
    fprintf('[%d/%d] 完成数据集: %s (%d个算法)\n', d, numel(dataset_names), ds, numel(methods));

    % --- 出图到 metrics_result/<dataset>/ ---
    out_dir = fullfile(output_root, ds);
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    % ε-切片
    pdf1 = fullfile(out_dir, sprintf('%s_epsilon_slice.pdf', ds));
    plot_epsilon_slice(methods, tau_grid, pdf1);
    % HV/IGD+ 条形图
    pdf2 = fullfile(out_dir, sprintf('%s_hv_igd_bar.pdf', ds));
    plot_bar_hv_igd(methods, ref_point, R, pdf2);
end

% ================= 统一输出所有结果 =================
fprintf('\n=================== 多目标评估结果汇总 ===================\n');
fprintf('指标说明: HV(↑) | IGD+(↓) | R2(↓) | 数据点数\n');
fprintf('(↑)越大越好  (↓)越小越好\n\n');

if ~isempty(all_results)
    % 创建表格数据
    num_results = length(all_results);
    dataset_names_all = cell(num_results, 1);
    algorithm_names_all = cell(num_results, 1);
    hv_values = zeros(num_results, 1);
    igd_values = zeros(num_results, 1);
    r2_values = zeros(num_results, 1);
    num_points = zeros(num_results, 1);
    
    for i = 1:num_results
        dataset_names_all{i} = all_results{i}.Dataset;
        algorithm_names_all{i} = all_results{i}.Algorithm;
        hv_values(i) = all_results{i}.HV;
        igd_values(i) = all_results{i}.IGDplus;
        r2_values(i) = all_results{i}.R2;
        num_points(i) = all_results{i}.NumPoints;
    end
    
    % 输出表格
    unique_datasets = unique(dataset_names_all, 'stable');
    unique_algorithms = unique(algorithm_names_all, 'stable');
    
    fprintf('%-20s %-12s %8s %8s %8s %6s\n', 'Dataset', 'Algorithm', 'HV(↑)', 'IGD+(↓)', 'R2(↓)', 'Points');
    fprintf('%s\n', repmat('-', 1, 70));
    
    for d = 1:length(unique_datasets)
        ds = unique_datasets{d};
        ds_indices = strcmp(dataset_names_all, ds);
        
        fprintf('%-20s\n', ds);
        for a = 1:length(unique_algorithms)
            algo = unique_algorithms{a};
            idx = find(ds_indices & strcmp(algorithm_names_all, algo));
            if ~isempty(idx)
                fprintf('%-20s %-12s %8.4f %8.4f %8.4f %6d\n', '', algo, ...
                    hv_values(idx), igd_values(idx), r2_values(idx), num_points(idx));
            end
        end
        fprintf('\n');
    end
    
    % 保存Excel文件
    excel_file = fullfile(output_root, 'metrics_summary.xlsx');
    result_table = table();
    result_table.Dataset = dataset_names_all';
    result_table.Algorithm = algorithm_names_all';
    result_table.HV_up = hv_values';
    result_table.IGDplus_down = igd_values';
    result_table.R2_down = r2_values';
    result_table.NumPoints = num_points';
    
    writetable(result_table, excel_file);
    fprintf('Excel结果已保存: %s\n', excel_file);
    
    % 最佳算法统计
    fprintf('\n=================== 最佳算法统计 ===================\n');
    for d = 1:length(unique_datasets)
        ds = unique_datasets{d};
        ds_indices = strcmp(dataset_names_all, ds);
        
        if sum(ds_indices) == 0, continue; end
        
        ds_hv = hv_values(ds_indices);
        ds_igd = igd_values(ds_indices);
        ds_r2 = r2_values(ds_indices);
        ds_algos = algorithm_names_all(ds_indices);
        
        [~, best_hv_idx] = max(ds_hv);
        [~, best_igd_idx] = min(ds_igd);
        [~, best_r2_idx] = min(ds_r2);
        
        fprintf('%-20s | HV最佳: %s(%.4f) | IGD+最佳: %s(%.4f) | R2最佳: %s(%.4f)\n', ...
            ds, ds_algos{best_hv_idx}, ds_hv(best_hv_idx), ...
            ds_algos{best_igd_idx}, ds_igd(best_igd_idx), ...
            ds_algos{best_r2_idx}, ds_r2(best_r2_idx));
    end
    
else
    fprintf('未收集到任何结果数据。\n');
end
fprintf('\n=================== 评估完成 ===================\n');

% --------------------------- 小工具 ------------------------------------
function p = relpath(target, base)
% 简单相对路径（仅为美观输出）
    try
        p = char(string(strrep(target, [base filesep], '')));
    catch
        p = target;
    end
end

function varargout = metrics(varargin)
% METRICS 入口（可选）。本文件包含多目标评测与绘图的全部子函数。
% 直接调用子函数即可，如：
%   Pnd = pareto_front_max(P);
%   hv  = hypervolume2d_max(P, [0,0]);
%   eps = epsilon_additive_max(A,B);
%   r2  = r2_indicator_max(P, generate_weights_2d(200), [1,1]); % R2 越小越好
%   igd = igd_plus_max(P, R);
%   plot_epsilon_slice(methods, tau_grid, save_pdf_path);
%   plot_bar_hv_igd(methods, ref, R, save_pdf_path);

% 仅为避免空文件警告
if nargout>0, varargout{1} = []; end
end

% ========================= 基本工具函数 ===============================

function Pnd = pareto_front_max(P)
% 返回 P 的最大化非支配点集合（去重）
    if isempty(P), Pnd = P; return; end
    P = unique(P,'rows');
    n = size(P,1);
    isDom = false(n,1);
    for i = 1:n
        if isDom(i), continue; end
        for j = 1:n
            if i==j, continue; end
            % j 支配 i（最大化）
            if all(P(j,:) >= P(i,:)) && any(P(j,:) > P(i,:))
                isDom(i) = true; break;
            end
        end
    end
    Pnd = P(~isDom,:);
end

function hv = hypervolume2d_max(P, ref)
% 二维最大化超体积（参考点在左下，劣于所有点），如 ref=[0,0]
    if isempty(P), hv = 0; return; end
    % 过滤掉不优于参考点的点（严格大于）
    keep = all(P > ref, 2);
    P = P(keep,:);
    if isempty(P), hv = 0; return; end

    P = pareto_front_max(P);
    [~, idx] = sort(P(:,1), 'descend'); % x 从大到小
    P = P(idx,:);
    ymono = cummax_safe(P(:,2));        % 强制单调，稳健

    x  = P(:,1);
    xn = [P(2:end,1); ref(1)];
    widths = max(x - xn, 0);
    hv = sum(widths .* ymono);
end

function c = cummax_safe(x)
% 兼容旧版 MATLAB 的 cummax
    c = x;
    for k = 2:numel(x)
        if c(k) < c(k-1)
            c(k) = c(k-1);
        end
    end
end

function eps_val = epsilon_additive_max(A, B)
% Additive ε 指标（最大化，越小越好）
% ε(A,B) = max_b min_a max_j (b_j - a_j)
    if isempty(A) || isempty(B), eps_val = NaN; return; end
    A = pareto_front_max(A); B = pareto_front_max(B);
    m = size(B,1);
    eps_each = zeros(m,1);
    for i = 1:m
        b = B(i,:);
        diffs = b - A;                   % 对每个 a
        worst_dim = max(diffs, [], 2);
        eps_each(i) = min(worst_dim);
    end
    eps_val = max(eps_each);
    eps_val = max(eps_val, 0);          % 数值稳健
end

function r2 = r2_indicator_max(P, W, zstar)
% R2 指标（最大化版的常用定义，**越小越好**）
% 使用加权切比雪夫 ASF 到理想点 z*（默认 [1,1]）
    if nargin < 3 || isempty(zstar), zstar = [1,1]; end
    if isempty(P), r2 = inf; return; end
    P = pareto_front_max(P);
    W = W ./ sum(W,2);                  % 归一化权重
    diffs = zstar - P;                  % |P| x d
    diffs = max(diffs, 0);              % 防止越界产生负值
    m = size(W,1);
    minASF = zeros(m,1);
    for i = 1:m
        w = W(i,:);
        asf_all = max(diffs .* w, [], 2);
        minASF(i) = min(asf_all);
    end
    r2 = mean(minASF);                  % 越小越好
end

function W = generate_weights_2d(M)
% 生成二维 L1 归一化权重（第一象限）
    if nargin<1, M=100; end
    theta = linspace(0, pi/2, M+2);
    theta = theta(2:end-1);
    W = [cos(theta(:)) sin(theta(:))];
    W = W ./ sum(W,2);
end

function igd = igd_plus_max(P, R)
% IGD+（最大化，越小越好）
    if isempty(P) || isempty(R), igd = NaN; return; end
    P = pareto_front_max(P);
    R = pareto_front_max(R);
    K = size(R,1);
    d = zeros(K,1);
    for i = 1:K
        r = R(i,:);
        delta = max(0, r - P);          % |P| x d
        dist  = sqrt(sum(delta.^2, 2));
        d(i)  = min(dist);
    end
    igd = mean(d);
end

% ========================= 评测与绘图封装 =============================

function T = compute_metrics_for_methods(methods, ref, zstar, W, R)
% 对一组方法（struct 数组：.name, .P (Nx2)）计算 HV/IGD+/R2
% 返回 table：method, HV, IGDplus, R2
    n = numel(methods);
    hv  = zeros(n,1);
    igd = zeros(n,1);
    r2  = zeros(n,1);
    for i = 1:n
        P = methods(i).P;
        Pnd = pareto_front_max(P);
        hv(i)  = hypervolume2d_max(Pnd, ref);
        igd(i) = igd_plus_max(Pnd, R);
        r2(i)  = r2_indicator_max(Pnd, W, zstar);
    end
    T = table(string({methods.name}'), hv, igd, r2, ...
        'VariableNames', {'Method','HV','IGDplus','R2'});
end

function [tau, bestACC] = epsilon_slice_max(methods, tau_grid)
% ε-切片（此处以第二指标 f_CCE 为阈值，取满足 f_CCE>=τ 时的最高 ACC）
% 返回：tau（列向量），bestACC（size: |tau| x |methods|）
    n = numel(methods);
    tau = tau_grid(:);
    bestACC = nan(numel(tau), n);
    for j = 1:n
        P = methods(j).P;
        acc  = P(:,1);          % 列1: ACC
        fcce = P(:,2);          % 列2: f_CCE
        for t = 1:numel(tau)
            mask = fcce >= tau(t);
            if any(mask)
                bestACC(t,j) = max(acc(mask));
            else
                bestACC(t,j) = NaN;
            end
        end
    end
end

function plot_epsilon_slice(methods, tau_grid, save_pdf_path)
% 绘制 ε-切片曲线并保存 PDF
    [tau, bestACC] = epsilon_slice_max(methods, tau_grid);
    f = figure('Visible','off'); hold on; box on;
    set(f,'Units','pixels','Position',[100 100 720 540]);
    cmap = lines(max(7,numel(methods)));
    for j = 1:numel(methods)
        plot(tau, bestACC(:,j), 'LineWidth', 1.8, 'Color', cmap(j,:));
    end
    xlabel('f\_CCE 阈值 \tau (越大越严格)');
    ylabel('在 f\_CCE \geq \tau 下的最佳 ACC');
    legend(string({methods.name}), 'Location','southwest','Interpreter','none');
    title('\epsilon-切片（以 f\_CCE 作约束）');
    set(gca,'FontName','Helvetica','FontSize',11);
    exportgraphics(f, save_pdf_path, 'ContentType','vector', 'BackgroundColor','white');
    close(f);
end

function plot_bar_hv_igd(methods, ref, R, save_pdf_path)
% 绘制 HV（↑）与 IGD+（↓）条形图（按 HV 降序排列）
    zstar = [1,1];
    W = generate_weights_2d(200);
    T = compute_metrics_for_methods(methods, ref, zstar, W, R);
    % 排序
    T = sortrows(T, 'HV', 'descend');

    f = figure('Visible','off'); set(f,'Units','pixels','Position',[100 100 760 520]);
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    % 左：HV
    nexttile;  bar(categorical(T.Method), T.HV);
    ylabel('HV（越大越好）'); title('Hypervolume');
    set(gca,'TickLabelInterpreter','none','FontName','Helvetica','FontSize',10);
    % 右：IGD+
    nexttile;  bar(categorical(T.Method), T.IGDplus);
    ylabel('IGD+（越小越好）'); title('IGD^+');
    set(gca,'TickLabelInterpreter','none','FontName','Helvetica','FontSize',10);

    exportgraphics(f, save_pdf_path, 'ContentType','vector', 'BackgroundColor','white');
    close(f);
end

% ======================== 数据提取小工具 ==============================

function P = extract_acc_fcce(data, path_str, type_str, prefix)
% 从不同的数据结构中抽取 [ACC, f_CCE] (两列, 值域[0,1])
% path_str 例如 'results.param_results'
% type_str in {'struct','cell'}
% prefix   例如 'avg_'（若字段是 avg_ACC, avg_f_CCE）
    % 取到容器
    container = data;
    if ~isempty(path_str)
        parts = strsplit(path_str, '.');
        for k = 1:numel(parts)
            if isfield(container, parts{k})
                container = container.(parts{k});
            else
                error('路径字段不存在：%s', parts{k});
            end
        end
    end

    % 扁平化为 cell of struct，便于遍历
    recs = {};
    switch lower(type_str)
        case 'cell'
            C = container;
            % 特殊处理：对于 iFairNMTF 和 KFC，如果是 cell 且第一个元素是 struct，则访问第一个元素
            if numel(C) >= 1 && isstruct(C{1})
                % 检查是否需要特殊处理
                first_elem = C{1};
                if isfield(first_elem, 'ACC') || isfield(first_elem, 'acc') || isfield(first_elem, 'f_CCE') || isfield(first_elem, 'fcce')
                    recs{end+1} = first_elem; %#ok<AGROW>
                else
                    % 常规处理
                    for i = 1:numel(C)
                        if isstruct(C{i}), recs{end+1} = C{i}; end %#ok<AGROW>
                    end
                end
            else
                for i = 1:numel(C)
                    if isstruct(C{i}), recs{end+1} = C{i}; end %#ok<AGROW>
                end
            end
        otherwise % 'struct'
            S = container;
            if isstruct(S)
               recs = num2cell(S); 
            else
                error('未知数据类型（非 struct/cell）');
            end
    end

    % 收集
    ACC = []; FCCE = [];
    for i = 1:numel(recs)
        r = recs{i};
        acc_name  = pick_field(r, {'ACC',[prefix 'ACC'],'acc',[prefix 'acc']});
        fcce_name = pick_field(r, {'f_CCE',[prefix 'f_CCE'],'F_CCE',[prefix 'F_CCE'],'fcce',[prefix 'fcce']});
        if isempty(acc_name) || isempty(fcce_name), continue; end
        a = r.(acc_name); f = r.(fcce_name);
        if isscalar(a) && isscalar(f)
            ACC  = [ACC; double(a)];
            FCCE = [FCCE; double(f)];
        elseif isvector(a) && isvector(f)
            n = min(numel(a), numel(f)); % 取较小的长度
            ACC  = [ACC; double(a(1:n))];  % 正确的索引方式
            FCCE = [FCCE; double(f(1:n))]; % 正确的索引方式
        end
    end
    P = [ACC, FCCE];
end

function name = pick_field(S, candidates)
% 在候选字段名中挑一个存在的
    name = '';
    for i = 1:numel(candidates)
        if isfield(S, candidates{i})
            name = candidates{i}; return;
        end
    end
end

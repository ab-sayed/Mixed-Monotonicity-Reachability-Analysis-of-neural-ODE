% % Paper:
% %   Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency
% %
% % Authors:  
% %   Abdelrahman Sayed Sayed, <abdelrahman.ibrahim -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
% %   Pierre-Jean Meyer, <pierre-jean.meyer -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
% %   Mohamed Ghazel, <mohamed.ghazel -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
% %
% % Date: 10th of August 2025
% % Last update: 17th of September 2025
% % Last revision: 17th of September 2025 
%
%------------- BEGIN CODE --------------

% User switch for plotting mode: 'full' for all reachable sets, 'final' for last set only
plot_mode = 'final'; % Options: 'final' or 'full'

dim = 2; % 2D for the Spiral

% Change default options
options.timeStep = 2 / 999; % Adjusted for 1000 time points ~ 0.01
options.taylorTerms = 2;
options.zonotopeOrder = 20;
options.alg = 'lin';
options.tensorOrder = 2;

% Initial states (2D)
x0 = [2.0; 0.0];
Initial_radius = 0.2;
lb = x0 - Initial_radius; % [1.8; -0.2]
ub = x0 + Initial_radius; % [2.2; 0.2]

R0 = interval(lb, ub);
U = zonotope([0, 0]);

params.R0 = zonotope(R0);
params.U = U;
params.timeStep = options.timeStep;
params.tFinal = 1;

% Verify time steps
steps = floor(params.tFinal / params.timeStep) + 1;
if steps ~= 1000
    warning('Expected 1000 time points, but computed %d. Adjusting timeStep.', steps);
    params.timeStep = params.tFinal / 999;
    options.timeStep = params.timeStep;
    steps = 1000;
end
disp(['Number of steps: ' num2str(steps)]);

% Define nonlinear system
sys = nonlinearSys(@spiral_non, 2, 1);

% =============================
% Compute full reachable set OA
% =============================
tic;
try
    R = reach(sys, params, options);
    disp('Full reachable set OA computed successfully.');
catch e
    disp('Error computing full reachable set OA:');
    disp(e.message);
    R = [];
end
fullSetTime = toc;
fprintf('Computation time for full reachable set OA: %.2f seconds\n', fullSetTime);

% =============================
% Compute Boundary reachable sets
% =============================
SplitsAll = 0; % We won't perform any splitting as we are not doing Safety
% Verification
SplitsB = 1;
count = 1;
intervalStep(1) = (ub(1) - lb(1)) / SplitsB;
intervalStep(2) = (ub(2) - lb(2)) / SplitsB;

R0B = cell(2 * dim, 1);
R0B_splits = cell(4 * SplitsB, 1);
R_Boundary = cell(4 * SplitsB, 1);
R_Boundary_times = zeros(4 * SplitsB, 1);
R_Boundary_volumes = zeros(4 * SplitsB, 1);

% Create upper boundary intervals
for i = 1:dim
    mintemp = R0.inf;
    maxtemp = R0.sup;
    mintemp(i) = R0.sup(i);
    R0B{i} = interval(mintemp, maxtemp);
    if i == 1
        lbsplit(1) = R0.sup(1);
        ubsplit(1) = R0.sup(1);
        for j = 1:SplitsB
            lbsplit(2) = lb(2) + (j-1) * intervalStep(2);
            ubsplit(2) = lb(2) + j * intervalStep(2);
            R0B_splits{count} = interval(lbsplit', ubsplit');
            count = count + 1;
        end
    else
        lbsplit(2) = R0.sup(2);
        ubsplit(2) = R0.sup(2);
        for j = 1:SplitsB
            lbsplit(1) = lb(1) + (j-1) * intervalStep(1);
            ubsplit(1) = lb(1) + j * intervalStep(1);
            R0B_splits{count} = interval(lbsplit', ubsplit');
            count = count + 1;
        end
    end
end

% Create lower boundary intervals
for i = 1:dim
    mintemp = R0.inf;
    maxtemp = R0.sup;
    maxtemp(i) = R0.inf(i);
    R0B{i + dim} = interval(mintemp, maxtemp);
    if i == 1
        lbsplit(1) = R0.inf(1);
        ubsplit(1) = R0.inf(1);
        for j = 1:SplitsB
            lbsplit(2) = lb(2) + (j-1) * intervalStep(2);
            ubsplit(2) = lb(2) + j * intervalStep(2);
            R0B_splits{count} = interval(lbsplit', ubsplit');
            count = count + 1;
        end
    else
        lbsplit(2) = R0.inf(2);
        ubsplit(2) = R0.inf(2);
        for j = 1:SplitsB
            lbsplit(1) = lb(1) + (j-1) * intervalStep(1);
            ubsplit(1) = lb(1) + j * intervalStep(1);
            R0B_splits{count} = interval(lbsplit', ubsplit');
            count = count + 1;
        end
    end
end

% Compute reachability for each boundary
for i = 1:4 * SplitsB
    current_interval = R0B_splits{i};
    if all(current_interval.inf == current_interval.sup)
        % Degenerate case: zero radius → small perturbation
        c = current_interval.inf;
        gen = 1e-6 * ones(dim, 1); % Small generator to avoid zero radius
        params.R0 = zonotope([c, gen]);
    else
        params.R0 = zonotope(current_interval);
    end

    try
        tic;
        R_Boundary{i} = reach(sys, params, options);
        R_Boundary_times(i) = toc;
        disp(['R_Boundary split ' num2str(i) ' computed successfully in ' ...
              num2str(R_Boundary_times(i)) ' seconds.']);
        if ~isempty(R_Boundary{i})
            R_Boundary_volumes(i) = volume(R_Boundary{i}.timePoint.set{end});
            disp(['R_Boundary split ' num2str(i) ' final zonotope volume: ' ...
                  num2str(R_Boundary_volumes(i))]);
        end
    catch e
        disp(['Error computing R_Boundary split ' num2str(i) ':']);
        disp(e.message);
        R_Boundary{i} = [];
        R_Boundary_times(i) = NaN;
        R_Boundary_volumes(i) = NaN;
    end
end

% Total computation time for boundary reachable sets (sum of individual times)
total_R_Boundary_time = sum(R_Boundary_times(~isnan(R_Boundary_times)));
fprintf('Total computation time for all R_Boundary splits: %.2f seconds\n', total_R_Boundary_time);

% =============================
% Save results for later analysis
% =============================
if ~isempty(R)
    succ_low_all_methods = zeros(dim, 1, 1);
    succ_up_all_methods = zeros(dim, 1, 1);
    current_zonotope = R.timePoint.set{end};
    current_interval = interval(current_zonotope);
    succ_low_all_methods(:, 1, 1) = current_interval.inf;
    succ_up_all_methods(:, 1, 1) = current_interval.sup;
else
    warning('Full reachable set R is empty. Using NaN placeholders.');
    succ_low_all_methods = nan(dim, 1, 1);
    succ_up_all_methods = nan(dim, 1, 1);
end

if ~isempty(R)
    full_boundary_vertices = vertices(R.timePoint.set{end});
else
    full_boundary_vertices = [];
end
boundary_splits_vertices = cell(4 * SplitsB, 1);
for i = 1:4 * SplitsB
    if ~isempty(R_Boundary{i})
        boundary_splits_vertices{i} = vertices(R_Boundary{i}.timePoint.set{end});
    else
        boundary_splits_vertices{i} = [];
    end
end

%% Set other variables for save compatibility of .mat file with other tools

t_points = linspace(0, params.tFinal, steps)';
n_x = dim;
n_methods = 1;
methods = [6]; % 'CORA Zonotope'

%% Generate 1000 sampled black points (not saved, only for plotting)
num_samples = 1000;
rand_succ = zeros(dim, num_samples);
for i = 1:num_samples
    x0_sample = lb + rand(dim,1) .* (ub - lb);
    [~, x_traj] = ode45(@(t, x) spiral_non(x, []), [0 params.tFinal], x0_sample);
    rand_succ(:,i) = x_traj(end,:)';
end

%% Plotting main figure with legend
matlab%% Plotting main figure with legend
figure('Name', 'Spiral Reachability Analysis - $x_1 - x_2$', 'Visible', 'on');
hold on;
plot_handles = [];
plot_labels = {};
if ~isempty(R) && strcmp(plot_mode, 'full')
    for j = 1:steps
        h = plot(R.timePoint.set{j}, [1, 2], 'r');
        if j == 1
            plot_handles(end+1) = h;
            plot_labels{end+1} = 'Full Reachable Set';
        end
    end
elseif ~isempty(R) && strcmp(plot_mode, 'final')
    h = plot(R.timePoint.set{end}, [1, 2], 'r', 'FaceAlpha', 0.3);
    plot_handles(end+1) = h;
    plot_labels{end+1} = 'Full Reachable Set (Final)';
end
for i = 1:SplitsAll^2
    if ~isempty(R_splits{i})
        h = plot(R_splits{i}.timePoint.set{end}, [1, 2], 'FaceColor', 'r', 'FaceAlpha', 0.3);
        if i == 1
            plot_handles(end+1) = h;
            plot_labels{end+1} = 'Uniform Splits';
        end
    end
end
allX = []; allY = [];
for i = 1:4 * SplitsB
    if ~isempty(R_Boundary{i})
        h = plot(R_Boundary{i}.timePoint.set{end}, [1, 2], 'FaceColor', 'b', 'FaceAlpha', 0.3);
        if i == 1
            plot_handles(end+1) = h;
            plot_labels{end+1} = 'Boundary only';
        end
        try
            V = vertices(R_Boundary{i}.timePoint.set{end});
            if ~isempty(V) && size(V, 1) >= 2
                v2 = V([1, 2], :);
                allX = [allX, v2(1, :)']; %#ok<AGROW>
                allY = [allY, v2(2, :)']; %#ok<AGROW>
            end
        catch e
            warning('Error extracting vertices for boundary split %d: %s\n', i, e.message);
        end
    end
end
K = []; % Initialize K to ensure it exists
if ~isempty(allX)
    try
        K = convhull(allX(:), allY(:));
        h_hull = plot(allX(K), allY(K), 'b--', 'LineWidth', 2);
        plot_handles(end+1) = h_hull;
        plot_labels{end+1} = 'Convex Hull of Boundaries';
    catch e
        warning('Convex hull computation failed: %s\n', e.message);
    end
end
h = plot(rand_succ(1,:), rand_succ(2,:), 'k.', 'MarkerSize', 8);
plot_handles(end+1) = h;
plot_labels{end+1} = 'Sampled Points';
legend(plot_handles, plot_labels, 'Location', 'best');
grid on;
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
title('$x_1$ vs. $x_2$', 'Interpreter', 'latex');
hold off;

%% Save main figure as EPS
[save_dir, save_name, ~] = fileparts(sprintf('reachability_data_sys_spiral_CORA_%s', datestr(now, 'yyyy-mm-dd_HH-MM-SS')));
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
main_fig_eps = fullfile(save_dir, [save_name '_main.eps']);
print('-depsc', main_fig_eps);
disp(['Main figure saved as ' main_fig_eps]);

%% Save individual zonotopes as EPS images
for i = 1:SplitsAll^2
    if ~isempty(R_splits{i})
        figure('Name', sprintf('Uniform Split %d Final Zonotope', i), 'Visible', 'off');
        hold on;
        plot(R_splits{i}.timePoint.set{end}, [1, 2], 'FaceColor', 'r', 'FaceAlpha', 0.3);
        grid on;
        xlabel('y_{1}');
        ylabel('y_{2}');
        title(sprintf('Uniform Split %d Final Zonotope', i));
        hold off;
        eps_file = fullfile(save_dir, sprintf('%s_R_split_%d_final.eps', save_name, i));
        print('-depsc', eps_file);
        disp(['Saved Uniform Split ' num2str(i) ' as ' eps_file]);
        close(gcf);
    end
end
for i = 1:4 * SplitsB
    if ~isempty(R_Boundary{i})
        figure('Name', sprintf('Boundary Split %d Final Zonotope', i), 'Visible', 'off');
        hold on;
        plot(R_Boundary{i}.timePoint.set{end}, [1, 2], 'FaceColor', 'b', 'FaceAlpha', 0.3);
        grid on;
        xlabel('y_{1}');
        ylabel('y_{2}');
        title(sprintf('Boundary Split %d Final Zonotope', i));
        hold off;
        eps_file = fullfile(save_dir, sprintf('%s_RB_split_%d_final.eps', save_name, i));
        print('-depsc', eps_file);
        disp(['Saved Boundary Split ' num2str(i) ' as ' eps_file]);
        close(gcf);
    end
end

%% Plot and save boundaries-only comparison figure
if ~isempty(R)
    figure('Name', 'Boundaries Only Comparison', 'Visible', 'on');
    hold on;
    h_full = plot(R.timePoint.set{end}, [1, 2], 'r-', 'LineWidth', 2, 'FaceAlpha', 0);
    allX_bound = []; allY_bound = [];
    for i = 1:4 * SplitsB
        if ~isempty(R_Boundary{i})
            h_bound = plot(R_Boundary{i}.timePoint.set{end}, [1, 2], 'b--', 'LineWidth', 1, 'FaceAlpha', 0);
            try
                V = vertices(R_Boundary{i}.timePoint.set{end});
                if ~isempty(V) && size(V, 1) >= 2
                    v2 = V([1, 2], :);
                    allX_bound = [allX_bound, v2(1, :)']; %#ok<AGROW>
                    allY_bound = [allY_bound, v2(2, :)']; %#ok<AGROW>
                end
            catch e
                warning('Error extracting vertices for boundary split %d in comparison: %s\n', i, e.message);
            end
        end
    end
    if ~isempty(allX_bound)
        try
            K_bound = convhull(allX_bound(:), allY_bound(:));
            h_hull_bound = plot(allX_bound(K_bound), allY_bound(K_bound), 'b--', 'LineWidth', 2);
        catch e
            warning('Convex hull computation failed in comparison: %s\n', e.message);
        end
    end
    grid on;
    xlabel('$x_1$', 'Interpreter', 'latex');
    ylabel('$x_2$', 'Interpreter', 'latex');
    title('OA Full Reachable Set vs Boundaries');
    legend([h_full, h_bound], {'Full Reachable Set OA', 'Boundaries only OA'}, 'Location', 'southeast');
    hold off;
    boundaries_eps = fullfile(save_dir, [save_name '_boundaries_only.eps']);
    print('-depsc', boundaries_eps);
    disp(['Boundaries-only comparison saved as ' boundaries_eps]);
end

%% Compute and display bounds for full reachable set
min_x_overall = inf(dim, 1);
max_x_overall = -inf(dim, 1);
if ~isempty(R)
    for i = 1:steps
        current_zonotope = R.timePoint.set{i};
        current_interval = interval(current_zonotope);
        lb_current = current_interval.inf;
        ub_current = current_interval.sup;
        min_x_overall = min(min_x_overall, lb_current);
        max_x_overall = max(max_x_overall, ub_current);
    end
    fprintf('\nOverall Reachable Tube Bounds:\n');
    fprintf('Lower bounds: [%s]\n', num2str(min_x_overall'));
    fprintf('Upper bounds: [%s]\n', num2str(max_x_overall'));
end

%% Tightness metric (area ratio) for full reachable set
if ~isempty(R)
    final_reachable_set = R.timePoint.set{end};
    final_interval = interval(final_reachable_set);
    lb_final = final_interval.inf;
    ub_final = final_interval.sup;
    zonotope_area_full = prod(ub_final - lb_final);
    actual_area = prod(max(rand_succ, [], 2) - min(rand_succ, [], 2));
    tightness_ratio_full = zonotope_area_full / actual_area;
    fprintf('\nTightness Metric (Area Ratio) for Full Reachable Set (CORA):\n');
    fprintf('Method CORA (Full): %.2f\n', tightness_ratio_full);
    fprintf('  Zonotope Area = %.6f, Actual Area = %.6f\n', zonotope_area_full, actual_area);
end

%% Tightness metric (area ratio) for boundaries-only reachable sets
if ~isempty(R_Boundary) && any(~cellfun(@isempty, R_Boundary))
    lb_bound = inf(dim, 1);
    ub_bound = -inf(dim, 1);
    for i = 1:4 * SplitsB
        if ~isempty(R_Boundary{i})
            current_zonotope = R_Boundary{i}.timePoint.set{end};
            current_interval = interval(current_zonotope);
            lb_bound = min(lb_bound, current_interval.inf);
            ub_bound = max(ub_bound, current_interval.sup);
        end
    end
    if all(isfinite(lb_bound)) && all(isfinite(ub_bound))
        zonotope_area_bound = prod(ub_bound - lb_bound);
        actual_area = prod(max(rand_succ, [], 2) - min(rand_succ, [], 2));
        tightness_ratio_bound = zonotope_area_bound / actual_area;
        fprintf('\nTightness Metric (Area Ratio) for Boundaries-Only Reachable Sets (CORA):\n');
        fprintf('Method CORA (Boundaries): %.2f\n', tightness_ratio_bound);
        fprintf('  Zonotope Area = %.6f, Actual Area = %.6f\n', zonotope_area_bound, actual_area);
    else
        warning('Unable to compute tightness metric for boundaries-only due to invalid intervals.');
    end
end

%% Save for comparison with other Tools & Methods
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
filename_base = sprintf('reachability_data_Boundary_sys_spiral_CORA_%s', timestamp);
boundary_hull_vertices = struct(); % New structure for convex hull boundaries
if ~isempty(allX) && ~isempty(allY) && exist('K', 'var') && ~isempty(K)
    hull_vertices = [allX(K)'; allY(K)']'; % Ensure 2xN by transposing column-wise
    boundary_hull_vertices.x1_x2 = hull_vertices;
    fprintf('Saved %d convex hull vertices for x1_x2 (2x%d matrix)\n', length(K), size(hull_vertices, 2));
elseif ~isempty(allX) && ~isempty(allY)
    hull_vertices = [allX'; allY']'; % Fallback to raw vertices as 2xN
    boundary_hull_vertices.x1_x2 = hull_vertices;
    warning('Convex hull indices empty, saved all %d raw vertices for x1_x2 (2x%d matrix)\n', length(allX), size(hull_vertices, 2));
else
    boundary_hull_vertices.x1_x2 = [];
    warning('No valid vertex data available for x1_x2 convex hull\n');
end
% Verify and display the saved matrix for debugging
if isfield(boundary_hull_vertices, 'x1_x2') && ~isempty(boundary_hull_vertices.x1_x2)
    disp('boundary_hull_vertices.x1_x2:');
    disp(boundary_hull_vertices.x1_x2); % Should show 2 rows
    fprintf('Matrix size: %dx%d\n', size(boundary_hull_vertices.x1_x2, 1), size(boundary_hull_vertices.x1_x2, 2));
end
save(filename_base, 'full_boundary_vertices', 'boundary_splits_vertices', 'boundary_hull_vertices', ...
    'succ_low_all_methods', 'succ_up_all_methods', 't_points', ...
    'methods', 'n_x', 'n_methods', '-v7.3');
disp(['Data saved to ' filename_base '.mat for boundary comparison']);

% %------------- END OF CODE --------------
% Paper:
%   Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency
%
% Authors:  
%   Abdelrahman Sayed Sayed, <abdelrahman.ibrahim -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
%   Pierre-Jean Meyer, <pierre-jean.meyer -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
%   Mohamed Ghazel, <mohamed.ghazel -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
%
% Date: 1st of August 2025
% Last update: 22nd of September 2025
% Last revision: 22nd of September 2025

%------------- BEGIN CODE --------------

function compute_spiral_reach(approach)
    % Handle missing input argument with a prompt or default
    if nargin < 1
        approach = input('Select approach (''initial'' or ''incremental''): ', 's');
        if isempty(approach)
            approach = 'initial'; % Default to initial if no input
        end
    end

    % Input validation for approach
    if ~ismember(approach, {'initial', 'incremental'})
        error('Approach must be ''initial'' or ''incremental''');
    end
    fprintf('Using approach: %s\n', approach);

    % Use MATLAB's default ode45 solver with default options
    ode_solver = @ode45;
    ode_options = odeset();
    fprintf('Using solver: ode45\n');

    % Parameters
    reachstep = 0.01; % Adjusted to ensure divisibility for the case of final_time=1 / 0.01
    initial_time = 0;
    intermediate_time = 0.5; % for the case of final_time=1
    final_time = 1;
    Initial_radius = 0.2;

    % Time points
    t_points = initial_time:reachstep:final_time;

    % Initial set (2D)
    x0 = [2.0; 0.0]; 
    lb = x0 - Initial_radius; % [1.8; -0.2]
    ub = x0 + Initial_radius; % [2.2; 0.2]
    init_set = Star(lb, ub);

    if strcmp(approach, 'initial')
        % Single-step reachability
        model = NonLinearODE(2, 1, @spiral_non, reachstep, final_time, eye(2));
        model.options.timeStep = reachstep;
        model.options.taylorTerms = 2;
        model.options.zonotopeOrder = 20;
        model.options.alg = 'lin';
        model.options.tensorOrder = 2;
        model = ODEblockLayer(model, final_time, reachstep, true);
        nn = NN({model});

        t = tic;
        R_tube = nn.reach(init_set);
        time = toc(t);
        fprintf('Reachability computation time: %.2f seconds\n', time);

        final_index = length(R_tube);
        fprintf('Length of R_tube: %d\n', final_index); % Debug output
        if isempty(R_tube) || final_index < 1
            error('Reachability computation failed: R_tube is empty');
        end
        final_star_set = R_tube(final_index);
        if ~isa(final_star_set, 'Star')
            error('final_star_set is not a Star object, type is: %s', class(final_star_set));
        end
        box_final = final_star_set.getBox;
        lb_final = box_final.lb;
        ub_final = box_final.ub;

    else % incremental
        % Incremental reachability
        model1 = NonLinearODE(2, 1, @spiral_non, reachstep, intermediate_time, eye(2));
        model1.options.timeStep = reachstep;
        model1.options.taylorTerms = 2;
        model1.options.zonotopeOrder = 20;
        model1.options.alg = 'lin';
        model1.options.tensorOrder = 2;
        model1 = ODEblockLayer(model1, intermediate_time, reachstep, true);
        nn1 = NN({model1});

        t = tic;
        R_tube1 = nn1.reach(init_set);
        time_phase1 = toc(t);
        fprintf('Phase 1 time: %.2f seconds\n', time_phase1);

        expected_steps = round(intermediate_time / reachstep) + 1;
        index_t1 = min(expected_steps, length(R_tube1));
        fprintf('Length of R_tube1: %d, index_t1: %d\n', length(R_tube1), index_t1); % Debug output
        if isempty(R_tube1) || index_t1 < 1
            error('Phase 1 reachability failed: R_tube1 is empty');
        end
        init_set_t1 = R_tube1(index_t1);

        model2 = NonLinearODE(2, 1, @spiral_non, reachstep, final_time - intermediate_time, eye(2));
        model2.options.timeStep = reachstep;
        model2.options.taylorTerms = 3;
        model2.options.zonotopeOrder = 10;
        model2.options.alg = 'lin';
        model2.options.tensorOrder = 2;
        model2 = ODEblockLayer(model2, final_time - intermediate_time, reachstep, true);
        nn2 = NN({model2});

        t = tic;
        R_tube2 = nn2.reach(init_set_t1);
        time_phase2 = toc(t);
        fprintf('Phase 2 time: %.2f seconds\n', time_phase2);

        final_index = length(R_tube2);
        fprintf('Length of R_tube2: %d\n', final_index); % Debug output
        if isempty(R_tube2) || final_index < 1
            error('Phase 2 reachability failed: R_tube2 is empty');
        end
        final_star_set = R_tube2(final_index);
        if ~isa(final_star_set, 'Star')
            error('final_star_set is not a Star object, type is: %s', class(final_star_set));
        end
        box_final = final_star_set.getBox;
        lb_final = box_final.lb;
        ub_final = box_final.ub;
    end

    % Store reachable sets
    n_t = length(t_points);
    n_x = 2;
    succ_low_all_methods = NaN(n_x, n_t);
    succ_up_all_methods = NaN(n_x, n_t);

    if strcmp(approach, 'initial')
        for k = 1:n_t
            if k <= length(R_tube)
                box = R_tube(k).getBox;
                succ_low_all_methods(:, k) = box.lb;
                succ_up_all_methods(:, k) = box.ub;
            end
        end
    else
        expected_steps_phase1 = round(intermediate_time / reachstep) + 1;
        for k = 1:expected_steps_phase1
            if k <= length(R_tube1)
                box = R_tube1(k).getBox;
                succ_low_all_methods(:, k) = box.lb;
                succ_up_all_methods(:, k) = box.ub;
            end
        end
        for k = (expected_steps_phase1 + 1):n_t
            idx = k - expected_steps_phase1;
            if idx <= length(R_tube2)
                box = R_tube2(idx).getBox;
                succ_low_all_methods(:, k) = box.lb;
                succ_up_all_methods(:, k) = box.ub;
            end
        end
    end

    % Monte Carlo sampling
    num_samples = 1000;
    final_states = zeros(2, num_samples);
    fprintf('Simulating %d samples using ode45...\n', num_samples);
    t_start = tic;
    for i = 1:num_samples
        x0_sample = lb + rand(2,1) .* (ub - lb);
        [~, x_traj] = ode_solver(@(t, x) spiral_non(x, zeros(1,1)), [initial_time final_time], x0_sample, ode_options);
        final_states(:,i) = x_traj(end,:)';
    end
    t_samples = toc(t_start);
    fprintf('Simulation time: %.2f seconds\n', t_samples);

    % Area computation (only one projection possible in 2D)
    % For full reachable set
    width = ub_final(1) - lb_final(1);
    height = ub_final(2) - lb_final(2);
    nnv_bounding_area = width * height;
    x_min = min(final_states(1, :));
    x_max = max(final_states(1, :));
    y_min = min(final_states(2, :));
    y_max = max(final_states(2, :));
    area_actual = (x_max - x_min) * (y_max - y_min);
    nnv_tightness_ratio = nnv_bounding_area / area_actual;

    fprintf('\nTightness Metric (Area Ratio) in 2D:\n');
    fprintf('NNV Star Set (Full Reachable Set): %.2f\n', nnv_tightness_ratio);
    fprintf('Full Reachable Set Bounding Box Area = %.6f, Actual Area = %.6f\n', nnv_bounding_area, area_actual);

    % Compute Boundary reachable sets
    SplitsB = 1; % Number of splits per dimension (CORA uses 1)
    dim = 2; % 2D system
    R0B_splits = cell(4 * SplitsB, 1); % 4 boundary sets (top, bottom, left, right)
    R_Boundary = cell(4 * SplitsB, 1); % Boundary reachable sets
    R_Boundary_times = zeros(4 * SplitsB, 1);
    R_Boundary_volumes = zeros(4 * SplitsB, 1);
    count = 1;

    % Define boundary intervals (similar to CORA)
    intervalStep = (ub - lb) / SplitsB; % Step size for each dimension
    % Upper boundaries
    for i = 1:dim
        lb_temp = lb;
        ub_temp = ub;
        lb_temp(i) = ub(i); % Upper boundary for dimension i
        ub_temp(i) = ub(i);
        if i == 1
            % x1 = ub(1), x2 varies
            lbsplit = [ub(1); lb(2)];
            ubsplit = [ub(1); ub(2)];
            R0B_splits{count} = Star(lbsplit, ubsplit);
            count = count + 1;
        else
            % x2 = ub(2), x1 varies
            lbsplit = [lb(1); ub(2)];
            ubsplit = [ub(1); ub(2)];
            R0B_splits{count} = Star(lbsplit, ubsplit);
            count = count + 1;
        end
    end
    % Lower boundaries
    for i = 1:dim
        lb_temp = lb;
        ub_temp = ub;
        lb_temp(i) = lb(i); % Lower boundary for dimension i
        ub_temp(i) = lb(i);
        if i == 1
            % x1 = lb(1), x2 varies
            lbsplit = [lb(1); lb(2)];
            ubsplit = [lb(1); ub(2)];
            R0B_splits{count} = Star(lbsplit, ubsplit);
            count = count + 1;
        else
            % x2 = lb(2), x1 varies
            lbsplit = [lb(1); lb(2)];
            ubsplit = [ub(1); lb(2)];
            R0B_splits{count} = Star(lbsplit, ubsplit);
            count = count + 1;
        end
    end

    % Compute reachability for each boundary set
    for i = 1:4 * SplitsB
        current_star = R0B_splits{i};
        if all(current_star.V(:,1) == current_star.V(:,2)) % Degenerate case
            % Add small perturbation to avoid zero radius
            center = current_star.V(:,1);
            gen = 1e-6 * ones(dim, 1);
            current_star = Star(center - gen, center + gen);
        end
        try
            tic;
            if strcmp(approach, 'initial')
                model = NonLinearODE(2, 1, @spiral_non, reachstep, final_time, eye(2));
                model.options.timeStep = reachstep;
                model.options.taylorTerms = 2;
                model.options.zonotopeOrder = 20;
                model.options.alg = 'lin';
                model.options.tensorOrder = 2;
                model = ODEblockLayer(model, final_time, reachstep, true);
                nn = NN({model});
                R_Boundary{i} = nn.reach(current_star);
            else % incremental
                model1 = NonLinearODE(2, 1, @spiral_non, reachstep, intermediate_time, eye(2));
                model1.options.timeStep = reachstep;
                model1.options.taylorTerms = 2;
                model1.options.zonotopeOrder = 20;
                model1.options.alg = 'lin';
                model1.options.tensorOrder = 2;
                model1 = ODEblockLayer(model1, intermediate_time, reachstep, true);
                nn1 = NN({model1});
                R_tube1 = nn1.reach(current_star);
                index_t1 = min(round(intermediate_time / reachstep) + 1, length(R_tube1));
                init_set_t1 = R_tube1(index_t1);
                model2 = NonLinearODE(2, 1, @spiral_non, reachstep, final_time - intermediate_time, eye(2));
                model2.options.timeStep = reachstep;
                model2.options.taylorTerms = 3;
                model2.options.zonotopeOrder = 10;
                model2.options.alg = 'lin';
                model2.options.tensorOrder = 2;
                model2 = ODEblockLayer(model2, final_time - intermediate_time, reachstep, true);
                nn2 = NN({model2});
                R_Boundary{i} = nn2.reach(init_set_t1);
            end
            R_Boundary_times(i) = toc;
            fprintf('R_Boundary split %d computed in %.2f seconds\n', i, R_Boundary_times(i));
            if ~isempty(R_Boundary{i})
                final_set = R_Boundary{i}(end);
                if isa(final_set, 'Star')
                    box = final_set.getBox;
                    if isa(box, 'Box')
                        R_Boundary_volumes(i) = prod(box.ub - box.lb);
                        fprintf('R_Boundary split %d final star set volume: %.6f\n', i, R_Boundary_volumes(i));
                    else
                        warning('Invalid box for R_Boundary split %d', i);
                        R_Boundary_volumes(i) = NaN;
                    end
                else
                    warning('Invalid final set for R_Boundary split %d', i);
                    R_Boundary_volumes(i) = NaN;
                end
            end
        catch e
            warning('Error computing R_Boundary split %d: %s', i, e.message);
            R_Boundary{i} = [];
            R_Boundary_times(i) = NaN;
            R_Boundary_volumes(i) = NaN;
        end
    end
    fprintf('Total computation time for all R_Boundary splits: %.2f seconds\n', sum(R_Boundary_times(~isnan(R_Boundary_times))));

    % Plotting main figure
    figure('Name', 'Spiral Reachability Analysis - $x_1 - x_2$');
    hold on;
    h1 = []; % Initialize handle for full reachable set
    if isa(final_star_set, 'Star')
        try
            fprintf('Attempting to plot Polyhedron with red boundaries\n');
            P = final_star_set.toPolyhedron;
            h1 = P.plot('edgecolor', 'r', 'alpha', 0, 'wire', true, 'linewidth', 2, 'DisplayName', 'Full Reachable Set'); % Red thicker boundaries
        catch e
            warning('Failed to plot star set: %s', e.message);
            % Fallback to manual box plotting
            try
                box = final_star_set.getBox;
                if isa(box, 'Box') && isvector(box.lb) && isvector(box.ub)
                    fprintf('Manually plotting box with red boundaries\n');
                    x = [box.lb(1) box.ub(1) box.ub(1) box.lb(1) box.lb(1)];
                    y = [box.lb(2) box.lb(2) box.ub(2) box.ub(2) box.lb(2)];
                    h1 = plot(x, y, 'r-', 'LineWidth', 2, 'DisplayName', 'Full Reachable Set'); % Manual red thicker rectangle
                else
                    warning('Invalid box for plotting');
                end
            catch e2
                warning('Fallback plotting failed: %s', e2.message);
            end
        end
    else
        warning('Cannot plot: final_star_set is not a valid Star object');
    end

    % Plot Monte Carlo sampled points
    h2 = plot(final_states(1, :), final_states(2, :), 'k.', 'MarkerSize', 5, 'DisplayName', 'Sampled points'); % Black dots

    xlabel('$x_1$', 'Interpreter', 'latex');
    ylabel('$x_2$', 'Interpreter', 'latex');
    title('$x_1$ vs. $x_2$', 'Interpreter', 'latex');
    grid on;
    if ~isempty(h1)
        legend([h1, h2], {'Full Reachable Set', 'Sampled points'}, 'Location', 'southeast'); % Legend with explicit handles
    else
        legend(h2, 'Sampled points', 'Location', 'southeast'); % Fallback legend
        warning('No boundaries plotted; legend shows only sampled points');
    end
    hold off;

    % Define timestamp and filename_base before plotting boundaries-only figure
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename_base = sprintf('reachability_data_Boundary_sys_spiral_NNV_%s_%s', approach, timestamp);

    % Plot boundaries-only comparison figure (full reachable set and convex hull)
    figure('Name', 'Full Reachable Set vs Convex Hull of Boundaries');
    hold on;
    h_full = []; % Handle for full reachable set
    if isa(final_star_set, 'Star')
        try
            P = final_star_set.toPolyhedron;
            h_full = P.plot('edgecolor', 'r', 'alpha', 0, 'wire', true, 'linewidth', 2, 'DisplayName', 'Full Reachable Set OA');
        catch e
            warning('Failed to plot full star set for comparison: %s', e.message);
            try
                box = final_star_set.getBox;
                if isa(box, 'Box') && isvector(box.lb) && isvector(box.ub)
                    x = [box.lb(1) box.ub(1) box.ub(1) box.lb(1) box.lb(1)];
                    y = [box.lb(2) box.lb(2) box.ub(2) box.ub(2) box.lb(2)];
                    h_full = plot(x, y, 'r-', 'LineWidth', 2, 'DisplayName', 'Full Reachable Set OA');
                else
                    warning('Invalid box for full set plotting');
                end
            catch e2
                warning('Fallback plotting failed for full set: %s', e2.message);
            end
        end
    end
    allX_bound = [];
    allY_bound = [];
    for i = 1:4 * SplitsB
        if ~isempty(R_Boundary{i})
            try
                final_set = R_Boundary{i}(end);
                if isa(final_set, 'Star')
                    P = final_set.toPolyhedron;
                    V = P.V;
                    if ~isempty(V) && size(V, 1) >= 2
                        allX_bound = [allX_bound; V(:, 1)]; % Collect x1 coordinates
                        allY_bound = [allY_bound; V(:, 2)]; % Collect x2 coordinates
                    end
                else
                    warning('Invalid final set for R_Boundary split %d', i);
                end
            catch e
                warning('Error extracting vertices for R_Boundary split %d: %s', i, e.message);
            end
        end
    end
    h_hull_bound = []; % Handle for convex hull
    if ~isempty(allX_bound)
        try
            K_bound = convhull(allX_bound, allY_bound);
            h_hull_bound = plot(allX_bound(K_bound), allY_bound(K_bound), 'b--', 'LineWidth', 2, 'DisplayName', 'Convex Hull of Boundaries');
            % Compute tightness metric for convex hull
            hull_x_min = min(allX_bound(K_bound));
            hull_x_max = max(allX_bound(K_bound));
            hull_y_min = min(allY_bound(K_bound));
            hull_y_max = max(allY_bound(K_bound));
            hull_bounding_area = (hull_x_max - hull_x_min) * (hull_y_max - hull_y_min);
            hull_tightness_ratio = hull_bounding_area / area_actual;
            fprintf('Convex Hull of Boundaries: %.2f\n', hull_tightness_ratio);
            fprintf('Convex Hull Bounding Box Area = %.6f, Actual Area = %.6f\n', hull_bounding_area, area_actual);
        catch e
            warning('Convex hull computation failed: %s', e.message);
        end
    end
    grid on;
    xlabel('$x_1$', 'Interpreter', 'latex');
    ylabel('$x_2$', 'Interpreter', 'latex');
    title('Full Reachable Set vs Convex Hull of Boundaries', 'Interpreter', 'latex');
    legend_handles = [];
    legend_labels = {};
    if ~isempty(h_full)
        legend_handles = [legend_handles, h_full];
        legend_labels{end+1} = 'Full Reachable Set OA';
    end
    if ~isempty(h_hull_bound)
        legend_handles = [legend_handles, h_hull_bound];
        legend_labels{end+1} = 'Convex Hull of Boundaries';
    end
    if ~isempty(legend_handles)
        legend(legend_handles, legend_labels, 'Location', 'southeast');
    else
        warning('No valid plot handles for comparison legend');
    end
    hold off;

    % Save boundaries-only comparison figure as EPS
    [save_dir, save_name, ~] = fileparts(filename_base);
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    boundaries_eps = fullfile(save_dir, [save_name '_boundaries_only.eps']);
    print('-depsc', boundaries_eps);
    disp(['Boundaries comparison (full set and convex hull) saved as ' boundaries_eps]);

    % Save for comparison with other Tools & Methods
    boundary_hull_vertices = struct(); % Structure for convex hull boundaries
    full_boundary_vertices = [];
    if isa(final_star_set, 'Star')
        try
            P = final_star_set.toPolyhedron;
            if ~isempty(P.V)
                full_boundary_vertices = P.V; % Full set vertices as Nx2 matrix
            end
        catch e
            warning('Failed to compute full boundary vertices: %s', e.message);
        end
    end
    if ~isempty(allX_bound) && ~isempty(allY_bound)
        try
            K = convhull(allX_bound, allY_bound);
            hull_vertices = [allX_bound(K)'; allY_bound(K)']'; % Ensure 2xN matrix
            boundary_hull_vertices.x1_x2 = hull_vertices;
            fprintf('Saved %d convex hull vertices for x1_x2 (2x%d matrix)\n', length(K), size(hull_vertices, 2));
        catch e
            warning('Convex hull computation failed: %s', e.message);
            boundary_hull_vertices.x1_x2 = [allX_bound'; allY_bound']'; % Fallback to raw vertices
            fprintf('Fallback: Saved %d raw vertices for x1_x2 (2x%d matrix)\n', length(allX_bound), size(boundary_hull_vertices.x1_x2, 2));
        end
    else
        boundary_hull_vertices.x1_x2 = [];
        warning('No valid vertex data available for x1_x2 convex hull');
    end
    % Verify and display the saved matrix for debugging
    if isfield(boundary_hull_vertices, 'x1_x2') && ~isempty(boundary_hull_vertices.x1_x2)
        disp('boundary_hull_vertices.x1_x2:');
        disp(boundary_hull_vertices.x1_x2); % Should show 2xN matrix
        fprintf('Matrix size: %dx%d\n', size(boundary_hull_vertices.x1_x2, 1), size(boundary_hull_vertices.x1_x2, 2));
    end
    save(filename_base, 'full_boundary_vertices', 'boundary_hull_vertices', '-v7.3');
    disp(['Data saved to ' filename_base '.mat for boundary comparison']);
end

%------------- END OF CODE --------------

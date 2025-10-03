% Paper:
%   Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency
%
% Authors:  
%   Abdelrahman Sayed Sayed, <abdelrahman.ibrahim -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
%   Pierre-Jean Meyer, <pierre-jean.meyer -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
%   Mohamed Ghazel, <mohamed.ghazel -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
%
% Date: 17th of July 2025
% Last update: 22nd of September 2025
% Last revision: 22nd of September 2025

%------------- BEGIN CODE --------------

function fpa_reach()

    % Use MATLAB's default ode45 solver with default options
    ode_solver = @ode45;
    ode_options = odeset();
    fprintf('Using solver: ode45\n');

    % User-defined approach selection
    % Set approach to 'initial' for single-step reachability or 'incremental' for two-phase reachability
    approach = 'initial'; % Options: 'initial' or 'incremental'

    % Parameters
    reachstep = 0.05; % Step size for reachability analysis: 0.05 "More loose - Used for Figure 9" or 2/500 "More Tight - Used for Table 5" 
    initial_time = 0; % Start time
    intermediate_time = 1; % Time to split phases (used only for incremental)
    final_time = 2; % End time
    Initial_radius = 0.01; % Uncertainty in initial conditions

    % Define time points
    t_points = initial_time:reachstep:final_time;

    % Define initial set at t=0
    x0 = [0, -0.58587, 0.8, 0.52323, 0.7]';
    lb = x0 - Initial_radius;
    ub = x0 + Initial_radius;
    init_set = Star(lb, ub);

    % Reachability analysis
    if strcmp(approach, 'initial')
        % Single-step reachability from t=0 to t=2
        model = NonLinearODE(5, 1, @CTRNN_FPA, reachstep, final_time, eye(5));
        model.options.timeStep = reachstep;
        model.options.taylorTerms = 3;
        model.options.zonotopeOrder = 5;
        model.options.alg = 'lin';
        model.options.tensorOrder = 2;
        model = ODEblockLayer(model, final_time, reachstep, true);
        nn = NN({model});

        t = tic;
        R_tube = nn.reach(init_set);
        time = toc(t);
        fprintf('Reachability computation time: %.2f seconds\n', time);

        % Final reachable set at t=2
        final_index = length(R_tube);
        final_star_set = R_tube(final_index);
        box_final = final_star_set.getBox;
        lb_final = box_final.lb;
        ub_final = box_final.ub;
    else % 'incremental'
        % Two-phase reachability
        % First phase: Model from t=0 to t=1
        model1 = NonLinearODE(5, 1, @CTRNN_FPA, reachstep, intermediate_time, eye(5));
        model1.options.timeStep = reachstep;
        model1.options.taylorTerms = 3;
        model1.options.zonotopeOrder = 10;
        model1.options.alg = 'lin';
        model1.options.tensorOrder = 2;
        model1 = ODEblockLayer(model1, intermediate_time, reachstep, true);
        nn1 = NN({model1});

        t = tic;
        R_tube1 = nn1.reach(init_set);
        time_phase1 = toc(t);
        fprintf('Phase 1 computation time (t=0 to t=1): %.2f seconds\n', time_phase1);

        % Extract reachable set at t=1
        expected_steps = round(intermediate_time / reachstep) + 1;
        if length(R_tube1) < expected_steps
            warning('R_tube1 length (%d) < expected (%d). Using last set.', length(R_tube1), expected_steps);
            index_t1 = length(R_tube1);
        else
            index_t1 = expected_steps;
        end
        init_set_t1 = R_tube1(index_t1);
        fprintf('Using Star set at index %d for t=1\n', index_t1);

        % Second phase: Model from t=1 to t=2
        model2 = NonLinearODE(5, 1, @CTRNN_FPA, reachstep, final_time - intermediate_time, eye(5));
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
        fprintf('Phase 2 computation time (t=1 to t=2): %.2f seconds\n', time_phase2);

        % Final reachable set at t=2
        final_index = length(R_tube2);
        final_star_set = R_tube2(final_index);
        box_final = final_star_set.getBox;
        lb_final = box_final.lb;
        ub_final = box_final.ub;
    end

    % Extract reachable sets at all time steps
    n_t = length(t_points);
    n_x = 5; % Number of states
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
    else % 'incremental'
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

    % Generate final states at t=final_time (simulating actual system behavior)
    num_samples = 1000; % Match TIRA's sample size
    final_states = zeros(5, num_samples);
    fprintf('Computing %d final states at t = %f using ode45...\n', num_samples, final_time);
    t_start = tic;
    for i = 1:num_samples
        x0_sample = lb + rand(5,1) .* (ub - lb); % Random initial states within bounds
        [~, x_traj] = ode_solver(@(t, x) CTRNN_FPA(x, zeros(1,1)), [initial_time final_time], x0_sample, ode_options);
        final_states(:,i) = x_traj(end,:)';
    end
    t_samples = toc(t_start);
    fprintf('Time to generate %d final states: %.2f seconds\n', num_samples, t_samples);

    % Define 2D projections (matching TIRA)
    dim_pairs = [1 2; 3 4; 4 5]; % x1-x2, x3-x4, x4-x5
    projections = {'x1-x2', 'x3-x4', 'x4-x5'};
    nnv_bounding_areas = zeros(3,1);
    area_actual = zeros(3,1);

    % Compute areas for each projection
    for p = 1:size(dim_pairs,1)
        dim1 = dim_pairs(p,1);
        dim2 = dim_pairs(p,2);
        % Bounding box area (NNV Star Set)
        width = ub_final(dim1) - lb_final(dim1);
        height = ub_final(dim2) - lb_final(dim2);
        nnv_bounding_areas(p) = width * height;
        % Actual spread area (simulated final states)
        x_min = min(final_states(dim1, :));
        x_max = max(final_states(dim1, :));
        y_min = min(final_states(dim2, :));
        y_max = max(final_states(dim2, :));
        area_actual(p) = (x_max - x_min) * (y_max - y_min);
    end

    % Compute tightness ratios
    nnv_tightness_ratios = nnv_bounding_areas ./ area_actual;

    % Display tightness metrics in a format similar to TIRA
    fprintf('\nTightness Metric (Area Ratio) for NNV Star Set method:\n');
    for p = 1:3
        fprintf('Projection %s:\n', projections{p});
        fprintf('Method NNV Star Set: %.2f\n', nnv_tightness_ratios(p));
        fprintf('  Bounding Box Area = %.6f, Area Actual = %.6f\n', nnv_bounding_areas(p), area_actual(p));
    end
    
    % Create directory for saving figures
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename_base = sprintf('reachability_data_Boundary_sys_FPA_NNV_%s_%s', approach, timestamp);
    [save_dir, ~, ~] = fileparts(filename_base);
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    % Plot the final star set for each projection with Monte Carlo samples
    for p = 1:size(dim_pairs,1)
        dim1 = dim_pairs(p,1);
        dim2 = dim_pairs(p,2);
        figure;
        hold on;
        h1 = []; % Initialize handle for boundaries
        if isa(final_star_set, 'Star') && ~isempty(final_star_set)
            try
                % Define projection matrix for 2D projection
                proj_mat = zeros(2, 5);
                proj_mat(1, dim1) = 1; % Project onto dim1
                proj_mat(2, dim2) = 1; % Project onto dim2
                proj_vec = zeros(2, 1); % No offset
                % Use affineMap to project the star set
                proj_star = final_star_set.affineMap(proj_mat, proj_vec);
                if ~isempty(proj_star) && ~proj_star.isEmptySet
                    % Convert to Polyhedron and plot with wireframe boundaries
                    P = proj_star.toPolyhedron;
                    if isa(P, 'Polyhedron')
                        fprintf('Plotting Polyhedron for projection %s with red boundaries\n', projections{p});
                        h1 = P.plot('edgecolor', 'r', 'alpha', 0, 'wire', true, 'linewidth', 2, 'DisplayName', 'Full Reachable Set OA'); % Red thicker boundaries
                    else
                        warning('Invalid Polyhedron for projection %s', projections{p});
                    end
                else
                    warning('Empty or invalid projection for final star set in projection %s', projections{p});
                end
            catch e
                warning('Failed to plot final star set for projection %s: %s', projections{p}, e.message);
                % Fallback to manual box plotting with wireframe
                try
                    box = final_star_set.getBox;
                    if isa(box, 'Box') && isvector(box.lb) && isvector(box.ub) && length(box.lb) >= max(dim1, dim2)
                        lb_2d = [box.lb(dim1); box.lb(dim2)];
                        ub_2d = [box.ub(dim1); box.ub(dim2)];
                        fprintf('Manually plotting box for projection %s with red boundaries\n', projections{p});
                        x = [lb_2d(1) ub_2d(1) ub_2d(1) lb_2d(1) lb_2d(1)];
                        y = [lb_2d(2) lb_2d(2) ub_2d(2) ub_2d(2) lb_2d(2)];
                        h1 = plot(x, y, 'r-', 'LineWidth', 2, 'DisplayName', 'Full Reachable Set OA'); % Manual red thicker rectangle
                    else
                        warning('Invalid box for projection %s', projections{p});
                    end
                catch e2
                    warning('Fallback plotting failed for projection %s: %s', projections{p}, e2.message);
                end
            end
        else
            warning('Invalid or empty final star set');
        end
        % Plot Monte Carlo sampled points for the final time
        h2 = plot(final_states(dim1, :), final_states(dim2, :), 'k.', 'MarkerSize', 5, 'DisplayName', 'Sampled points'); % Black dots
        title(sprintf('$x_%d$ vs. $x_%d$', dim1, dim2), 'Interpreter', 'latex'); % Updated LaTeX title
        xlabel(sprintf('$x_%d$', dim1), 'Interpreter', 'latex'); % Updated LaTeX x-label
        ylabel(sprintf('$x_%d$', dim2), 'Interpreter', 'latex'); % Updated LaTeX y-label
        grid on;
        if ~isempty(h1)
            legend([h1, h2], {'Full Reachable Set OA', 'Sampled points'}, 'Location', 'southeast');
        else
            legend(h2, 'Sampled points', 'Location', 'best'); % Fallback legend if boundaries not plotted
            warning('No boundaries plotted for projection %s; legend shows only sampled points', projections{p});
        end
        % Save figure as EPS and PNG
        fig_filename_base = fullfile(save_dir, sprintf('%s_%s', filename_base, projections{p}));
        print(fig_filename_base, '-depsc'); % Save as EPS
        print(fig_filename_base, '-dpng'); % Save as PNG
        fprintf('Saved figure for projection %s as %s.eps and %s.png\n', projections{p}, fig_filename_base, fig_filename_base);
        hold off;
    end

    %% Save for comparison with other Tools & Methods
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename_base = sprintf('reachability_data_Boundary_sys_FPA_NNV_%s_%s', approach, timestamp);
    boundary_hull_vertices = struct(); % New structure for convex hull boundaries
    for p = 1:size(dim_pairs,1)
        dim1 = dim_pairs(p,1);
        dim2 = dim_pairs(p,2);
        field_name = sprintf('x%d_x%d', dim1, dim2);
        if isa(final_star_set, 'Star')
            try
                % Define projection matrix for 2D projection
                proj_mat = zeros(2, 5);
                proj_mat(1, dim1) = 1;
                proj_mat(2, dim2) = 1;
                proj_vec = zeros(2, 1);
                proj_star = final_star_set.affineMap(proj_mat, proj_vec);
                P = proj_star.toPolyhedron;
                if ~isempty(P.V)
                    boundary_hull_vertices.(field_name) = P.V; % Vertices as Nx2 matrix
                    fprintf('Saved %d convex hull vertices for %s (%dx%d matrix)\n', size(P.V, 1), field_name, size(P.V, 1), size(P.V, 2));
                else
                    boundary_hull_vertices.(field_name) = [];
                    warning('No vertices available for %s convex hull\n', field_name);
                end
            catch e
                warning('Failed to compute Polyhedron vertices for %s: %s', field_name, e.message);
                % Fallback to box vertices
                try
                    box = final_star_set.getBox;
                    if isa(box, 'Box') && isvector(box.lb) && isvector(box.ub)
                        hull_vertices = [
                            box.lb(dim1), box.lb(dim2);
                            box.ub(dim1), box.lb(dim2);
                            box.ub(dim1), box.ub(dim2);
                            box.lb(dim1), box.ub(dim2)
                        ];
                        boundary_hull_vertices.(field_name) = hull_vertices;
                        fprintf('Fallback: Saved 4 box vertices for %s (4x2 matrix)\n', field_name);
                    else
                        boundary_hull_vertices.(field_name) = [];
                        warning('Invalid box for %s vertices\n', field_name);
                    end
                catch e2
                    boundary_hull_vertices.(field_name) = [];
                    warning('Fallback failed for %s vertices: %s', field_name, e2.message);
                end
            end
        else
            boundary_hull_vertices.(field_name) = [];
            warning('No valid Star set for computing %s vertices\n', field_name);
        end
        % Verify and display the saved matrix for debugging
        if isfield(boundary_hull_vertices, field_name) && ~isempty(boundary_hull_vertices.(field_name))
            disp(['boundary_hull_vertices.' field_name ':']);
            disp(boundary_hull_vertices.(field_name)); % Should show Nx2 matrix
            fprintf('Matrix size: %dx%d\n', size(boundary_hull_vertices.(field_name), 1), size(boundary_hull_vertices.(field_name), 2));
        end
    end
    save(filename_base, 'boundary_hull_vertices', ...
         'succ_low_all_methods', 'succ_up_all_methods', 't_points', 'n_x', '-v7.3');

    disp(['Data saved to ' filename_base '.mat for boundary comparison']);
end

%------------- END OF CODE --------------

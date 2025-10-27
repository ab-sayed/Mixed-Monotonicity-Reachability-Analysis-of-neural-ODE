% Paper:
%   Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency
%
% Authors:  
%   Abdelrahman Sayed Sayed, <abdelrahman.ibrahim -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
%   Pierre-Jean Meyer, <pierre-jean.meyer -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
%   Mohamed Ghazel, <mohamed.ghazel -AT- univ-eiffel.fr>, COSYS-ESTAS, Univ Gustave Eiffel
%
% Date: 11th of August 2025
% Last update: 27th of October 2025
% Last revision: 17th of September 2025

%------------- BEGIN CODE --------------

% User switch for plotting mode: 'full' for all reachable sets, 'final' for last set only
plot_mode = 'final'; % 'full' or 'final'

dim = 5;

% Change default options
options.timeStep    = 2 / 999;     % ~1000 time points over t in [0,1]
options.taylorTerms = 3;
options.zonotopeOrder = 5;         % higher may crash for this model
options.alg = 'lin';
options.tensorOrder = 2;

% Initial states (5D)
x0 = [0, -0.58587, 0.8, 0.52323, 0.7]';  % CTRNN_FPA nominal
Initial_radius = 0.01;
lb = x0 - Initial_radius;
ub = x0 + Initial_radius;

R0 = interval(lb, ub);
U  = zonotope([0, 0]);

params.R0     = zonotope(R0);
params.U      = U;
params.timeStep = options.timeStep;
params.tFinal   = 2;  % 1 or 2 based on preference and results from TIRA and NNV 2.0

% Verify time steps (target = 1000)
steps = floor(params.tFinal / params.timeStep) + 1;
if steps ~= 1000
    warning('Expected 1000 time points, computed %d. Adjusting timeStep.', steps);
    params.timeStep = params.tFinal / 999;
    options.timeStep = params.timeStep;
    steps = 1000;
end
disp(['Number of time points: ' num2str(steps)]);

% Define nonlinear system
% Make sure CTRNN_FPA(x,u) is on path and returns xdot for CORA's nonlinearSys wrapper.
sys = nonlinearSys(@CTRNN_FPA, dim, 1);

%% === Full reachable set OA ===
t_all_start = tic;
full_start = tic;
try
    R = reach(sys, params, options);
    fullSetTime = toc(full_start);
    fprintf('Computation time (Full OA): %.3f s\n', fullSetTime);
catch e
    disp('Error computing Full OA:'); disp(e.message);
    R = []; fullSetTime = NaN;
end

%% === Boundary (corner) reachable sets: 2^5 = 32 ===
% Corner k corresponds to a binary pattern across dims:
% bit=0 -> lower boundary in that dim (fix at lb), bit=1 -> upper boundary (fix at ub).
nSplits = 2^dim;
binPatterns = dec2bin(0:nSplits-1) - '0';   % [32 x 5] matrix of 0/1

RB_splits        = cell(nSplits, 1);
RB_splits_times  = zeros(nSplits, 1);
RB_splits_final  = cell(nSplits, 1);        % final time zonotopes
labelsLU         = strings(nSplits,1);      % e.g. "LULUU"

boundary_total_start = tic;
for k = 1:nSplits
    lbsplit = lb; ubsplit = ub;
    pat = binPatterns(k,:);  % row vector of length 5

    % Build L/U label and fix each dim to its chosen boundary
    lu_chars = repmat('L', 1, dim);
    for d = 1:dim
        if pat(d) == 0  % lower boundary in dim d
            ubsplit(d) = lb(d);
            lu_chars(d) = 'L';
        else            % upper boundary in dim d
            lbsplit(d) = ub(d);
            lu_chars(d) = 'U';
        end
    end
    labelsLU(k) = string(lu_chars);

    params.R0 = zonotope(interval(lbsplit, ubsplit));
    try
        one_start = tic;
        RB_splits{k} = reach(sys, params, options);
        RB_splits_times(k) = toc(one_start);

        if ~isempty(RB_splits{k})
            RB_splits_final{k} = RB_splits{k}.timePoint.set{end};
            fprintf('RB_split %2d (%s) computed in %.3f s\n', k, labelsLU(k), RB_splits_times(k));
        else
            RB_splits_final{k} = [];
            fprintf('RB_split %2d (%s) is empty.\n', k, labelsLU(k));
        end
    catch e
        disp(['Error computing RB_split ' num2str(k) ' (' char(labelsLU(k)) '):']);
        disp(e.message);
        RB_splits{k} = []; RB_splits_final{k} = []; RB_splits_times(k) = NaN;
    end
end
boundary_total_time = toc(boundary_total_start);
fprintf('Total computation time (all 32 boundary splits): %.3f s\n', boundary_total_time);

total_pipeline_time = toc(t_all_start);
fprintf('Whole pipeline time (Full + Boundaries): %.3f s\n', total_pipeline_time);

%% === Prepare output directories and color map ===
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
outDir = fullfile(pwd, ['FPA_Results_' timestamp]);
figDir = fullfile(outDir, 'figures');
indivDir = fullfile(figDir, 'individual_splits');
comboDir = fullfile(figDir, 'combined');

if ~exist(outDir, 'dir'), mkdir(outDir); end
if ~exist(figDir, 'dir'), mkdir(figDir); end
if ~exist(indivDir, 'dir'), mkdir(indivDir); end
if ~exist(comboDir, 'dir'), mkdir(comboDir); end

% 32 distinct-ish colors
cmap = lines(nSplits);    % fallback; works fine and is stable
% Alternatively: cmap = hsv(nSplits);

%% === Helper: safe extraction of 2D vertices for a zonotope ===
% Tries vertices(); if it fails/is empty, falls back to interval corners.
get2DVertices = @(Z, i1, i2) ...
    local_get2DVertices(Z, i1, i2);

%% === Full set vertices (final) for reference (optional) ===
if ~isempty(R)
    try
        fullV = vertices(R.timePoint.set{end});  % 5D vertices (may be large)
    catch
        fullV = []; % not essential; we plot the patch directly anyway
    end
else
    fullV = [];
end

%% === Projections to plot ===
dim_pairs = [1 2; 3 4; 4 5];
proj_names = {'x1_x2', 'x3_x4', 'x4_x5'}; % Changed hyphens to underscores

%% === Save mapping legend (index, L/U pattern, RGB) ===
mapFile = fullfile(outDir, 'mapping_legend.txt');
fid = fopen(mapFile, 'w');
fprintf(fid, 'SplitIndex\tPattern(L/U per dim [1..5])\tColorRGB\n');
for k = 1:nSplits
    fprintf(fid, '%02d\t\t%s\t\t[%.3f %.3f %.3f]\n', k, labelsLU(k), cmap(k,1), cmap(k,2), cmap(k,3));
end
fclose(fid);
disp(['Mapping legend saved: ' mapFile]);

%% === Combined figures: full OA (red) + all 32 colored boundary OAs + convex hull ===
% Also compute & overlay convex hull of union of all boundary vertices for each projection.

% % Modified to store the Convex Hulls of the Boundary reachability
hulls = struct; % store hull indices & points per projection

for p = 1:size(dim_pairs,1)
    d1 = dim_pairs(p,1); d2 = dim_pairs(p,2);
    allX = []; allY = [];

    f = figure('Visible','on'); hold on; box on; grid on;
    parts = split(proj_names{p}, '_');
    title_str = sprintf('Boundaries of Full Reachable Set OA vs Boundaries only OA $x_{%s} - x_{%s}$', parts{1}(2:end), parts{2}(2:end));
    title(title_str, 'Interpreter', 'latex');

    % Plot full OA first and assign handle
    h_full = [];
    if ~isempty(R)
        try
            h_full = plot(R.timePoint.set{end}, [d1 d2], ...
                'r-', 'LineWidth', 4, 'FaceAlpha', 0);
        catch
            h_full = plot(R.timePoint.set{end}, [d1 d2]);
        end
    end

    % Plot boundary OAs and collect vertices
    h_boundary = patch(nan, nan, [0 0 1], 'FaceAlpha', 0.3, 'EdgeColor', 'b', 'LineWidth', 1.5, 'Visible', 'off');
    num_valid_vertices = 0;
    for k = 1:nSplits
        if isempty(RB_splits_final{k})
            fprintf('Split %d (%s) is empty.\n', k, labelsLU(k));
            continue;
        end
        try
            [vx, vy] = get2DVertices(RB_splits_final{k}, d1, d2);
            if ~isempty(vx)
                patch(vx, vy, cmap(k,:), 'FaceColor', cmap(k,:), 'FaceAlpha', 0.3, ...
                      'EdgeColor', cmap(k,:)*0.6, 'LineWidth', 1.5);
                allX = [allX, vx(:)']; 
                allY = [allY, vy(:)'];
                num_valid_vertices = num_valid_vertices + length(vx);
                fprintf('Split %d (%s): %d vertices in projection [%d,%d]\n', ...
                        k, labelsLU(k), length(vx), d1, d2);
            else
                warning('No vertices for split %d projection [%d,%d]\n', k, d1, d2);
            end
        catch e
            fprintf('Error plotting split %d (%s): %s\n', k, labelsLU(k), e.message);
            plot(RB_splits_final{k}, [d1 d2]); % Fallback
        end
    end
    fprintf('Projection %s: Collected %d total vertices\n', proj_names{p}, num_valid_vertices);

    % % Plot sampled black points --> [Not working] Un-comment to allow the plotting of the random
    % % sampled black points
    %
    % if exist('rand_succ', 'var') && ~isempty(rand_succ)
    %     h_points = plot(rand_succ(d1,:), rand_succ(d2,:), 'k.', 'MarkerSize', 8);
    %     if p == 1
    %         plot_handles(end+1) = h_points;
    %         plot_labels{end+1} = 'Sampled Points';
    %     end
    % else
    %     warning('rand_succ not available or empty for plotting sampled points.');
    % end

    % Plot convex hull and assign handle
    h_hull = [];
    if ~isempty(allX)
        try
            K = convhull(allX(:), allY(:));
            if ~isempty(K)
                h_hull = plot(allX(K), allY(K), 'b--', 'LineWidth', 2);
                hulls(p).K = K;
                hulls(p).X = allX;
                hulls(p).Y = allY;
                fprintf('Projection %s: %d vertices in convex hull\n', proj_names{p}, length(K));
            else
                warning('Convex hull returned empty indices for projection %s.\n', proj_names{p});
                hulls(p).K = [];
                hulls(p).X = allX;
                hulls(p).Y = allY;
            end
        catch e
            warning('Convex hull failed for projection %s: %s\n', proj_names{p}, e.message);
            hulls(p).K = [];
            hulls(p).X = allX;
            hulls(p).Y = allY;
        end
    else
        warning('No boundary vertices collected for projection %s.\n', proj_names{p});
        hulls(p).K = [];
        hulls(p).X = [];
        hulls(p).Y = [];
    end

    % Create legend with proper handles
    legend_entries = {'Full Reachable Set', 'Sampled Points', 'Convex Hull of Boundaries'};
    handles = [h_full, h_hull];
    valid_handles = handles(arrayfun(@(x) ~isempty(x) && isgraphics(x), handles));
    if ~isempty(valid_handles)
        n_handles = length(valid_handles);
        legend(valid_handles, legend_entries(1:n_handles), 'Location','southeast');
    else
        warning('No valid handles for legend in projection %s.', proj_names{p});
    end

    xlabel(sprintf('$x_%d$', d1), 'Interpreter', 'latex');
    ylabel(sprintf('$x_%d$', d2), 'Interpreter', 'latex');

    % Adjust axis limits with padding
    if ~isempty(R)
        V_full = vertices(R.timePoint.set{end});
        if ~isempty(V_full)
            x_min = min(V_full(d1,:)); x_max = max(V_full(d1,:));
            y_min = min(V_full(d2,:)); y_max = max(V_full(d2,:));
        else
            I_full = interval(R.timePoint.set{end});
            x_min = I_full.inf(d1); x_max = I_full.sup(d1);
            y_min = I_full.inf(d2); y_max = I_full.sup(d2);
        end
    else
        x_min = inf; x_max = -inf; y_min = inf; y_max = -inf;
    end
    if ~isempty(allX)
        x_min = min(x_min, min(allX)); x_max = max(x_max, max(allX));
        y_min = min(y_min, min(allY)); y_max = max(y_max, max(allY));
    end
    if exist('rand_succ', 'var') && ~isempty(rand_succ)
        x_min = min(x_min, min(rand_succ(d1,:))); x_max = max(x_max, max(rand_succ(d1,:)));
        y_min = min(y_min, min(rand_succ(d2,:))); y_max = max(y_max, max(rand_succ(d2,:)));
    end
    x_range = x_max - x_min; y_range = y_max - y_min;
    buffer = 0.1;
    axis([x_min - buffer*x_range, x_max + buffer*x_range, y_min - buffer*y_range, y_max + buffer*y_range]);

    % Save PNG and EPS
    fn_png = fullfile(comboDir, sprintf('Full_vs_Boundary_withHull_%s.png', proj_names{p}));
    print(f, fn_png, '-dpng', '-r300');
    fn_eps = fullfile(comboDir, sprintf('Full_vs_Boundary_withHull_%s.eps', proj_names{p}));
    print(f, fn_eps, '-depsc');
end
disp(['Saved combined figures in: ' comboDir]);

%% === Compute Tightness Metrics ===
% Generate 1000 sampled points for actual area estimation
num_samples = 1000;
rand_succ = zeros(dim, num_samples);
for i = 1:num_samples
    x0_sample = lb + rand(dim,1) .* (ub - lb);
    [~, x_traj] = ode45(@(t, x) CTRNN_FPA(x, []), [0 params.tFinal], x0_sample);
    rand_succ(:,i) = x_traj(end,:)';
end

% Tightness metric for full reachable set
if ~isempty(R)
    fprintf('\nTightness Metric (Area Ratio) for Full Reachable Set (FPA):\n');
    for p = 1:length(dim_pairs)
        d1 = dim_pairs(p,1); d2 = dim_pairs(p,2);
        final_zonotope = R.timePoint.set{end};
        final_interval = interval(final_zonotope);
        lb_final = [final_interval.inf(d1), final_interval.inf(d2)];
        ub_final = [final_interval.sup(d1), final_interval.sup(d2)];
        zonotope_area_full = prod(ub_final - lb_final);
        actual_area = prod(max(rand_succ([d1, d2],:), [], 2) - min(rand_succ([d1, d2],:), [], 2));
        tightness_ratio_full = zonotope_area_full / actual_area;
        fprintf('Projection %s (FPA Full): %.2f\n', proj_names{p}, tightness_ratio_full);
        fprintf('  Zonotope Area = %.6f, Actual Area = %.6f\n', zonotope_area_full, actual_area);
    end
end

% Tightness metric for boundaries-only reachable sets
if ~any(cellfun(@isempty, RB_splits_final))
    fprintf('\nTightness Metric (Area Ratio) for Boundaries-Only Reachable Sets (FPA):\n');
    for p = 1:length(dim_pairs)
        d1 = dim_pairs(p,1); d2 = dim_pairs(p,2);
        lb_bound = inf(1, 2); ub_bound = -inf(1, 2);
        for k = 1:nSplits
            if ~isempty(RB_splits_final{k})
                current_zonotope = RB_splits_final{k};
                current_interval = interval(current_zonotope);
                lb_bound = min(lb_bound, [current_interval.inf(d1), current_interval.inf(d2)]);
                ub_bound = max(ub_bound, [current_interval.sup(d1), current_interval.sup(d2)]);
            end
        end
        if all(isfinite(lb_bound)) && all(isfinite(ub_bound))
            zonotope_area_bound = prod(ub_bound - lb_bound);
            actual_area = prod(max(rand_succ([d1, d2],:), [], 2) - min(rand_succ([d1, d2],:), [], 2));
            tightness_ratio_bound = zonotope_area_bound / actual_area;
            fprintf('Projection %s (FPA Boundaries): %.2f\n', proj_names{p}, tightness_ratio_bound);
            fprintf('  Zonotope Area = %.6f, Actual Area = %.6f\n', zonotope_area_bound, actual_area);
        else
            warning('Unable to compute tightness metric for projection %s due to invalid intervals.', proj_names{p});
        end
    end
end

%% === Save Data for Comparison with Other Tools and Methods ===

% % Modified to store the convex hull of the boundaries
full_boundary_vertices = struct();
boundary_splits_vertices = cell(nSplits, 1);
boundary_hull_vertices = struct(); % New structure for convex hull boundaries

for p = 1:length(dim_pairs)
    d1 = dim_pairs(p,1); d2 = dim_pairs(p,2);
    if ~isempty(R)
        try
            V = vertices(R.timePoint.set{end});
            if ~isempty(V) && size(V, 1) >= max(d1, d2)
                full_boundary_vertices.(proj_names{p}) = V([d1, d2], :);
            else
                full_boundary_vertices.(proj_names{p}) = [];
            end
        catch e
            full_boundary_vertices.(proj_names{p}) = [];
        end
    else
        full_boundary_vertices.(proj_names{p}) = [];
    end
    for k = 1:nSplits
        if ~isempty(RB_splits_final{k})
            try
                V = vertices(RB_splits_final{k}.timePoint.set{end});
                if ~isempty(V) && size(V, 1) >= max(d1, d2)
                    boundary_splits_vertices{k} = V([d1, d2], :);
                else
                    boundary_splits_vertices{k} = [];
                end
            catch e
                boundary_splits_vertices{k} = [];
            end
        else
            boundary_splits_vertices{k} = [];
        end
    end
    % Store convex hull boundaries or raw vertices if hull fails
    if isstruct(hulls) && p <= numel(hulls) && ~isempty(hulls(p).X) && ~isempty(hulls(p).Y)
        if ~isempty(hulls(p).K)
            hull_vertices = [hulls(p).X(hulls(p).K); hulls(p).Y(hulls(p).K)];
            boundary_hull_vertices.(proj_names{p}) = hull_vertices;
            fprintf('Projection %s: Saved %d convex hull vertices\n', proj_names{p}, length(hulls(p).K));
        else
            % Fallback: save all collected vertices if hull indices are empty
            hull_vertices = [hulls(p).X; hulls(p).Y];
            boundary_hull_vertices.(proj_names{p}) = hull_vertices;
            warning('Projection %s: Convex hull indices empty, saved all %d raw vertices\n', proj_names{p}, length(hulls(p).X));
        end
    else
        boundary_hull_vertices.(proj_names{p}) = [];
        warning('Projection %s: No valid hull or vertex data available\n', proj_names{p});
    end
end

% Placeholder for success metrics and methods
succ_low_all_methods = zeros(length(dim_pairs), nSplits);
succ_up_all_methods = zeros(length(dim_pairs), nSplits);
t_points = linspace(0, params.tFinal, steps)';
methods = {'FPA'};
n_x = dim;
n_methods = length(methods);

% Save to .mat file
filename_base = fullfile(outDir, ['FPA_Comparison_' datestr(now, 'yyyy-mm-dd_HH-MM-SS')]);
save(filename_base, 'full_boundary_vertices', 'boundary_splits_vertices', 'boundary_hull_vertices', ...
    'succ_low_all_methods', 'succ_up_all_methods', 't_points', ...
    'methods', 'n_x', 'n_methods', '-v7.3');
disp(['Data saved to ' filename_base '.mat for boundary comparison']);

%% ================= Local helper (at end of file) =================
function [vx, vy] = local_get2DVertices(Z, i1, i2)
%LOCAL_GET2DVERTICES  Robustly obtain 2D vertex set (vx, vy) for a zonotope Z projected on [i1,i2]
% 1) Try CORA's vertices(Z) and project.
% 2) If that fails/empty, fall back to interval corners.
    vx = []; vy = [];
    % if isempty(Z), return; end
    if isempty(Z), disp('Z is empty'); return; end % debug to see if it is falling
    try
        V = vertices(Z);   % size [dim x nV]
        disp(['Vertices size: ' num2str(size(V))]);
        if ~isempty(V)
            v2 = V([i1,i2], :);
            vx = v2(1,:); vy = v2(2,:);
            return;
        end
    catch e
        disp(['Vertices error: ' e.message]);
        % fallthrough
    end
    % Fallback: interval corners (very coarse)
    try
        I = interval(Z);
        lb2 = [I.inf(i1); I.inf(i2)];
        ub2 = [I.sup(i1); I.sup(i2)];
        C = [lb2, [lb2(1);ub2(2)], [ub2(1);lb2(2)], ub2]; % 4 rectangle corners
        vx = C(1,:); vy = C(2,:);
    catch
        % give up
        vx = []; vy = [];
    end
end


% %------------- END OF CODE --------------
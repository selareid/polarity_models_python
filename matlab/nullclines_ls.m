
% Parameters
clear variables;
base_params = parameters();
params_ls = base_params;
params_ls.rho_A = 0.5 * base_params.rho_A;
params_ls.rho_P = 0.15 * base_params.rho_P;

% Define symbolic variables and nullclines
syms xM yA; % we use xm and ya so that they are plotted in the right order
[nc_rj, nc_ra, asym_acyto, asym_jcyto] = nullcline_defns(xM, yA, params_ls);


%% Plot full region of interest

figure(1)
clf;
hold on;

region_MA = [-0.5 1.2*params_ls.rho_J/params_ls.psi -0.5 1.2*params_ls.rho_A/params_ls.psi];

nc_rj_sol = fimplicit(nc_rj, region_MA, 'MeshDensity',1000);
nc_ra_sol = fimplicit(nc_ra, region_MA, 'MeshDensity',1000);

asym_jcyto_sol = fimplicit( asym_jcyto, region_MA, '--k', 'MeshDensity',500);
fimplicit(asym_acyto, region_MA, 'w', 'LineWidth', 2);
asym_acyto_sol = fimplicit(asym_acyto, region_MA, '--k');
fimplicit(asym_jcyto, region_MA, '--k');

yline(0, '--k')
xline(0, '--k')

% Formatting
xlabel('m');
ylabel('a');
xlim(region_MA(1:2))
ylim(region_MA(3:4))
legend('r_j=0', 'r_a=0', "Domain edges")


%% Plot subregion of interest

figure(2)
clf('reset');
hold on;
subregion_MA = [0 1.5 -0.5 1];
fimplicit(nc_rj, subregion_MA, 'MeshDensity',5000);
fimplicit(nc_ra, subregion_MA, 'MeshDensity',5000);

% Formatting
xlabel('M');
ylabel('A');
xlim(subregion_MA(1:2))
ylim(subregion_MA(3:4))
legend('rP=0', 'rM=0')


%% Save data
% Convert to table format
make_table_sol = @(sol) make_variable_table(sol.XData', sol.YData', params_ls);
nc_rj_data_ls = make_table_sol(nc_rj_sol);
nc_ra_data_ls = make_table_sol(nc_ra_sol);
asym_acyto_data_ls = make_table_sol(asym_acyto_sol);
asym_jcyto_data_ls = make_table_sol(asym_jcyto_sol);

% Export
writetable(nc_rj_data_ls);
writetable(nc_ra_data_ls);
writetable(asym_acyto_data_ls);
writetable(asym_jcyto_data_ls);


%% Find steady state values and the associated J/P values as well
% sols = solve([nc_rj, nc_ra], [xM, yA]);
guess = [0.7, 0.4];
sols = vpasolve([nc_rj, nc_ra], [xM, yA], guess);
steady_states_ls = steady_states_analysis(sols, params_ls)

% Write to file
writetable(steady_states_ls)


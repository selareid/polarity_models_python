% Parameters
clear variables;
params = parameters();

% Define symbolic variables and nullclines
syms xM yA; % we use xm and ya so that they are plotted in the right order
[nc_rj, nc_ra, asym_acyto, asym_jcyto] = nullcline_defns(xM, yA, params);


%% Plot full region of interest

figure(1)
clf;
hold on;

region_MA = [-0.5 1.2*params.rho_J/params.psi -0.5 1.2*params.rho_A/params.psi];

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
subregion_MA = [0 1.2 0 0.6];
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
make_table_sol = @(sol) make_variable_table(sol.XData', sol.YData', params);
nc_rj_data = make_table_sol(nc_rj_sol);
nc_ra_data = make_table_sol(nc_ra_sol);
asym_acyto_data = make_table_sol(asym_acyto_sol);
asym_jcyto_data = make_table_sol(asym_jcyto_sol);

% Export
writetable(nc_rj_data);
writetable(nc_ra_data);
writetable(asym_acyto_data);
writetable(asym_jcyto_data);


%% Find steady state values and stability
% sols = solve([nc_rj, nc_ra], [xM, yA]);
guess = [0.3; 0.03];
digits(50)
sols = vpasolve([nc_rj, nc_ra], [xM, yA], guess);
steady_states = steady_states_analysis(sols, params)

% Write to file
writetable(steady_states)

% Check reaction term values if getting a warning from vpasolve
% fh_rj = matlabFunction(lhs(nc_rj), 'Vars', [xM yA]);
% fh_ra = matlabFunction(lhs(nc_ra), 'Vars', [xM yA]);
% for i = 1:height(steady_states)
%     steady_states(i,:)
%     M = steady_states.M(i);
%     A = steady_states.A(i);
%     rJ = fh_rj(M, A)
%     rA = fh_ra(M, A)
% end

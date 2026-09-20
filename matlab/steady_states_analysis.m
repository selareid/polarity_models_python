function steady_states = steady_states_analysis(sols, params)

M_sols = double(sols.xM);
A_sols = double(sols.yA);

% Just want the real solutions
real_idx = imag(M_sols) == 0 & imag(A_sols) == 0;
M_sols_real = M_sols(real_idx);
A_sols_real = A_sols(real_idx);

% Make into a table
steady_states = make_variable_table(M_sols_real, A_sols_real, params);

n_ss = height(steady_states);
steady_states.stability = repmat(["stable"],n_ss,1);
steady_states.eigenvalues = NaN(n_ss,4);
for idx = 1:n_ss
    % steady_states(idx,[4, 1, 2, 3])
    Jac = get_jacobian(steady_states.J(idx), steady_states.M(idx), ...
                    steady_states.A(idx), steady_states.P(idx), params);
    
    eigenvalues = eig(Jac)';
    steady_states.eigenvalues(idx, :) = eigenvalues;
    % Check if any of eigenvalues is positive
    if any(eigenvalues > 0)
        steady_states.stability(idx) = "unstable";
    end
end

end


function Jac = get_jacobian(J, M, A, P, parms)
    % grad of f' w.r.t the different species
    A_cyto = parms.rho_A-parms.psi*(A+M);
    dAcyto_dA = -parms.psi; dAcyto_dM = -parms.psi;
    dJcyto_dJ = -parms.psi; dJcyto_dM = -parms.psi;
    dPcyto_dP = -parms.psi;

    grad_J = [
        -parms.konM*A_cyto + parms.konJ*dJcyto_dJ - parms.koffJ - parms.kJP*P, % dj'/dj
        -parms.konM*dAcyto_dM*J + parms.kdisM + parms.konJ*dJcyto_dM,  % dj'/dm
        -parms.konM*dAcyto_dA*J, % dj'/da
        -parms.kJP*J % dj'/dp
    ]';
    grad_M = [
        parms.konM*A_cyto, % dm'/dj
        parms.konM*dAcyto_dM*J - parms.kdisM - parms.koffM - parms.kMP*P, % dm'/dm
        parms.konM*dAcyto_dA*J, % dm'/da
        -parms.kMP * M % dm'/dp
    ]';
    grad_A = [
        0, % da'/dj
        parms.kdisM, % da'/dm
        -parms.koffA - parms.kAP*P, % da'/da
        -parms.kAP*A % da'/dp
    ]';
    grad_P = [
        0, % dp'/dj
        -2*parms.kPA*(A+M)*P, % dp'/dm
        -2*parms.kPA*(A+M)*P, % dp'/da
        parms.konP*dPcyto_dP - parms.koffP - parms.kPA*(A+M)^2
    ]';
    Jac = [grad_J; grad_M; grad_A; grad_P];
end
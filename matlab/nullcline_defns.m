function [nc_rj, nc_ra, asym_acyto, asym_jcyto]  ...
                                            = nullcline_defns(M, A, params)
    
    % Functions for P(M,A) and J(M,A, P(M,A))
    A_cyto = params.rho_A - params.psi*(A+M);
    P = params.rho_P*params.konP/(params.psi*params.konP + params.koffP + params.kPA*(A+M)^2);
    J = (params.kdisM*M+params.koffM*M+params.kMP*P*M)/(params.konM*A_cyto);
    J_cyto = params.rho_J - params.psi*(J + M);
    
    % Define the reaction equations for j and a as a function of m and a
    RJ = -params.konM*A_cyto*J + params.kdisM*M + params.konJ*J_cyto - params.koffJ*J-params.kJP*P*J;
    RA = params.kdisM*M - params.koffA*A - params.kAP*P*A;
    
    % Now define the nullclines and asymptotes to solve for
    nc_rj = (RJ == 0);
    nc_ra = (RA == 0);
    asym_acyto = (A_cyto == 0);
    asym_jcyto = (J_cyto == 0);
    
end
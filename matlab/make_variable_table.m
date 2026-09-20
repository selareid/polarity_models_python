function output_table = make_variable_table(m, a, params)
    output_table = table(m, a,  'VariableNames', ["M", "A"]);
    
    a_cyto = a_cyto_fn(m, a, params);
    p = p_fn(m, a, params);
    j = j_fn(m, p, a_cyto, params);
    j_cyto = j_cyto_fn(j, m, params);
    
    output_table.P = p;
    output_table.J = j;
    output_table.A_cyto = a_cyto;
    output_table.J_cyto = j_cyto;
end

function P = p_fn(m, a, params)
    P = params.rho_P .* params.konP ./ (params.psi .* params.konP + params.koffP + params.kPA .* (a+m).^2);
end


function J = j_fn(m, p, a_cyto, params)
    J = (params.kdisM .* m + params.koffM .* m + params.kMP .* p .* m) ./ (params.konM .* a_cyto);
end

function A_cyto = a_cyto_fn(m, a, params)
    A_cyto = params.rho_A - params.psi .* (a+m);
end

function J_cyto = j_cyto_fn(j, m, params)
    J_cyto = params.rho_J - params.psi*(j + m);
end

from typing import Callable, Optional
import numpy as np
from dataclasses import dataclass, field
from scipy import integrate

def default_v_func(k: Parameters, x, t):
    v_time = 600
    time_scaling = 1 / np.maximum(1, t / 10 - v_time / 10)
    c = 1/4
    s = c / 4
    peak = 0.1/67.3
    return time_scaling * peak * np.exp(-(x - c) ** 2 / (2 * s ** 2))

# No advection case (for maintenance/loss)
def zero_v_func(k: Parameters, x, t):
    return 0

@dataclass(frozen=True) # To be safe don't let this be modified after initialisation as could mess with X and deltax
class Parameters:
    Species: tuple[str, ...] = ("J", "M", "A", "P") # Use tuple instead of list for immutability
    
    # General space specification variables
    Nx: int = 101
    # L: float = 134.6
    x0: int = 0
    # xL: float = 67.3
    xL: float = 1.0

    # Time specification variables
    t0: float = 0
    tL: float =  1e4
    points_per_second: float = 0.01

    # Advection function
    v_func: Callable = default_v_func
    
    # Model parameters
    psi: float = 0.174

    # Diffusion parameters
    D_J: float = 0.28 / (67.3**2)   # D_J / (xL**2), 
    D_M: float = 7.5e-2 / (67.3**2) # D_M / (xL**2),
    D_A: float = 0.28 / (67.3**2)   # D_A / (xL**2),
    D_P: float = 0.15 / (67.3**2)   # D_P / (xL**2),
    
    konM: float = 9.01e-3
    kdisM: float = 1.64e-3
    
    kJP: float = 6.16e-2
    kMP: float = 4.41e-2
    kAP: float = 4.61e-1
    kPA: float = 2
    
    rho_J: float = 1.2
    rho_A: float = 1.56
    rho_P: float = 1.0
    
    konJ: float = 1.4e-2
    konP: float = 4.74e-2
    
    koffJ: float = 1.17e-3
    koffM: float = 8.44e-3
    koffA: float = 2.65e-3
    koffP: float = 7.3e-3
    
    # Not used
    sigmaJ: float = 1
    sigmaM: float = 1
    sigmaP: float = 1
    
    konA: float = 0
    alpha: int = 1
    beta: int = 2

    # Need to do a little bit of processing for some of the parameters
    X: np.ndarray = field(init = False) # Not set by user
    deltax: float = field(init = False) # Not set by user
    initial_condition: Optional[list] = None # Can be set by user, otherwise default is set in post init function
    t_eval: Optional[np.ndarray] = None # Can be set by user, otherwise default is set in post init function

    def __post_init__(self):
        # We need to use object.__setattr__ since the dataclass is frozen
        object.__setattr__(self, 'X', np.linspace(self.x0, self.xL, self.Nx))
        object.__setattr__(self, 'deltax', np.abs(self.X[1] - self.X[0]))
        if self.initial_condition is None:
            object.__setattr__(self, 'initial_condition', ( [1] * (self.Nx*3) + [0]*self.Nx ) )
        if self.t_eval is None:
            t_eval = np.linspace(self.t0, self.tL, 
                                int(self.points_per_second * np.abs(self.tL - self.t0)) + 1)
            object.__setattr__(self, 't_eval', t_eval)

DEFAULT_PARAMETERS = Parameters()


# Diffusion and advection solving functions
#------------------------------------------------------------------------------------------
def disc_diffusion_term(k: Parameters, Y, x_i):
    # This function accounts for boundary reflection
    if x_i == 0:  # left boundary
        return (Y[1] - 2 * Y[0] + Y[1]) / k.deltax ** 2  # reflect Y[-1] to Y[1]
    elif x_i == k.Nx - 1:  # right boundary
        return (Y[k.Nx - 2] - 2 * Y[k.Nx - 1] + Y[k.Nx - 2]) / k.deltax ** 2  # reflect Y[Nx] over Nx-1 to Y[Nx-2]
    else:  # internal point
        return (Y[x_i + 1] - 2 * Y[x_i] + Y[x_i - 1]) / k.deltax ** 2


# where func is a function of type x_i -> float
def disc_spatial_derivative(k: Parameters, func: Callable[[int], float], x_i):
    return (func(x_i + 1) - func(x_i)) / k.deltax


# Reaction functions and cytoplasm concentrations
#------------------------------------------------------------------------------------------
# Reaction functions
def R_J(k: Parameters, J, M, A, P, t, x_i, A_cyto_r, J_cyto_r):
    return (-k.konM*A_cyto_r*J[x_i] + k.kdisM*M[x_i] \
           + k.konJ*J_cyto_r - k.koffJ*J[x_i] \
            - k.kJP*P[x_i]**k.alpha*J[x_i])
            # - k.kJP*(P[x_i]+M[x_i])**k.alpha*J[x_i])

def R_M(k: Parameters, J, M, A, P, t, x_i, A_cyto_r):
    return (k.konM*A_cyto_r*J[x_i] - k.kdisM*M[x_i] \
           - k.koffM*M[x_i] \
           - k.kMP*P[x_i]*M[x_i])

def R_A(k: Parameters, J, M, A, P, t, x_i, A_cyto_r):
    return (k.kdisM*M[x_i] + k.konA*A_cyto_r - k.koffA*A[x_i] \
           - k.kAP*P[x_i]*A[x_i])

def R_P(k: Parameters, J, M, A, P, t, x_i, P_cyto_r):
    return (k.konP*P_cyto_r - k.koffP*P[x_i] \
           - k.kPA*(A[x_i]+M[x_i])**k.beta*P[x_i])


# Cytoplasmic concentrations
def J_cyto(k:Parameters, J, M): 
    return k.rho_J - k.psi * Ybar(k, J) - k.psi * Ybar(k, M)

def A_cyto(k:Parameters, A, M): 
    return k.rho_A - k.psi * Ybar(k, A) - k.psi * Ybar(k, M)

def P_cyto(k:Parameters, P): return k.rho_P - k.psi * Ybar(k, P)

def Ybar(k: Parameters, Y):
    return integrate.simpson(Y, x=k.X) / k.xL  # all of J,A,P-bar
    # 2 * integrate.simpson(Y, x = k["X"]) / k["L"]  # all of J,A,P-bar


# ODE system function
#------------------------------------------------------------------------------------------
# State variable order: J, M, A, P
def odefunc(t, U, k: Parameters):
    Nx = k.Nx

    assert len(U) == 4 * Nx

    # Failure so odefunc doesn't run forever trying to fix numerical issues
    if min(U) < -100 or max(U) > 100:
        print(f"FAILURE at simulation time {t:.4f}")
        # plot_failure(U, t, k)
        raise AssertionError

    J = U[:Nx]
    M = U[Nx:2*Nx]
    A = U[2*Nx:3*Nx]
    P = U[3*Nx:]

    dudt_J = [0]*Nx
    dudt_M = [0]*Nx
    dudt_A = [0]*Nx
    dudt_P = [0]*Nx

    # r is for "resolved"
    J_cyto_r = J_cyto(k, J, M)
    A_cyto_r = A_cyto(k, A, M)
    P_cyto_r = P_cyto(k, P)

    # insides
    # diffusion function handles left boundary
    for x_i in np.arange(0, Nx-1):
        dudt_J[x_i] = k.D_J*disc_diffusion_term(k, J, x_i) \
                        -k.sigmaJ*disc_spatial_derivative(k, lambda x_ii: k.v_func(k, k.X[x_ii], t)*J[x_ii], x_i) \
                        + R_J(k, J, M, A, P, t, x_i, A_cyto_r, J_cyto_r)
        dudt_M[x_i] = k.D_M*disc_diffusion_term(k, M, x_i) \
                        -k.sigmaM*disc_spatial_derivative(k, lambda x_ii: k.v_func(k, k.X[x_ii], t)*M[x_ii], x_i) \
                        + R_M(k, J, M, A, P, t, x_i, A_cyto_r)
        dudt_A[x_i] = k.D_A*disc_diffusion_term(k, A, x_i) \
                        + R_A(k, J, M, A, P, t, x_i, A_cyto_r)
        dudt_P[x_i] = k.D_P*disc_diffusion_term(k, P, x_i) \
                        -k.sigmaP*disc_spatial_derivative(k, lambda x_ii: k.v_func(k, k.X[x_ii], t)*P[x_ii], x_i) \
                        + R_P(k, J, M, A, P, t, x_i, P_cyto_r)

    # manually handle right boundary ( x_i = Nx-1 ) since v(x,t) is odd
    # reflect Nx over Nx-1 to Nx-2; for v_func, also negate on the reflection as v(x)=-v(-x)
    x_i = Nx-1
    dudt_J[x_i] = k.D_J*disc_diffusion_term(k, J, x_i) \
                    - (-k.v_func(k, k.X[Nx-2], t)*J[Nx-2] - k.v_func(k, k.X[Nx-1], t)*J[Nx-1]) / k.deltax \
                    + R_J(k, J, M, A, P, t, x_i, A_cyto_r, J_cyto_r)
    dudt_M[x_i] = k.D_M*disc_diffusion_term(k, M, x_i) \
                    - (-k.v_func(k, k.X[Nx-2], t)*M[Nx-2] - k.v_func(k, k.X[Nx-1], t)*M[Nx-1]) / k.deltax \
                    + R_M(k, J, M, A, P, t, x_i, A_cyto_r)
    dudt_A[x_i] = k.D_A*disc_diffusion_term(k, A, x_i) \
                    + R_A(k, J, M, A, P, t, x_i, A_cyto_r)
    dudt_P[x_i] = k.D_P*disc_diffusion_term(k, P, x_i) \
                    - (-k.v_func(k, k.X[Nx-2], t)*P[Nx-2] - k.v_func(k, k.X[Nx-1], t)*P[Nx-1]) / k.deltax \
                    + R_P(k, J, M, A, P, t, x_i, P_cyto_r)

    return dudt_J + dudt_M + dudt_A + dudt_P


# To solve the model --------------------------------------------------------------------------------------
def run_model(args={}, calc_ss_initial_condition=False):

    if calc_ss_initial_condition:
        args["initial_condition"] = run_for_ss_initial_condition(args)
    
    # Update any parameters with passed args, otherwise we use defaults
    kvals = Parameters(**args)

    # Solve
    sol = integrate.solve_ivp(odefunc, 
                              [kvals.t0, kvals.tL], kvals.initial_condition, method="BDF",
                              t_eval=kvals.t_eval, args=(kvals,))

    return sol, kvals


# Function to get the steady state initial condition -------------------------------------------------------
def run_for_ss_initial_condition(args = {}) -> list:
    # Timings
    output_times = [0, 2000]
    tL = output_times[-1]
    sol0, _ = run_model({**args,
        "t_eval": output_times, "tL": tL,
        "v_func": zero_v_func
    })
    return sol0.y[:,-1]


def run_for_polarised_initial_condition(args = {}) -> list:
    # Get the homogeneous steady state initial condition first
    start_initial_condition = run_for_ss_initial_condition(args)
    # Timings
    ic_tL = 120*60
    sol0, _ = run_model({
        **args,
        "t_eval": [0, ic_tL],
        "tL": ic_tL,
        "initial_condition": start_initial_condition
    })
    return sol0.y[:,-1]
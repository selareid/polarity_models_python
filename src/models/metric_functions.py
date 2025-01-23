import numpy as np
from scipy import integrate


# Analysis Metric
# A,P just on membrane
def polarity_measure(X, Am, Pm, Nx):
    measure, _, _ = polarity_get_all(X, Am, Pm, Nx)
    return measure


# determine the orientation of polarity
# 0 - undetermined
# 1 - anterior on left
# 2 - anterior on right
def polarity_orientation(X, Am, Pm, Nx):
    _, orientation, _ = polarity_get_all(X, Am, Pm, Nx)
    return orientation


def orientation_marker(orientation_code):
    return ['o', '<', '>'][orientation_code]


# returns (measure, orientation, marker)
def polarity_get_all(X, Am, Pm, Nx):
    a_left = integrate.simpson(Am[:Nx//2], x = X[:Nx//2])
    a_right = integrate.simpson(Am[Nx//2:], x = X[Nx//2:])
    p_left = integrate.simpson(Pm[:Nx//2], x = X[:Nx//2])
    p_right = integrate.simpson(Pm[Nx//2:], x = X[Nx//2:])

    measure = 0 if ((a_left + a_right)*(p_left + p_right)) == 0 else np.abs(a_left - a_right) * np.abs(p_left - p_right) / ((a_left + a_right)*(p_left + p_right))

    # Orientation
    if a_left > a_right and p_right > p_left:  # A is on the left, P is on the right
        orientation = 1
    elif a_left < a_right and p_right < p_left:  # A is on the right, P is on the left
        orientation = 2
    else:  # undetermined
        orientation = 0

    return measure, orientation, orientation_marker(orientation)


# polarity metric but just using the posterior quantity
# def posterior_polarity_get_all(X, Pm, Nx):
#     assert Nx > 3  # no meaningful information when step-size too low
#
#     p_left = integrate.simpson(Pm[:Nx//2], X[:Nx//2])
#     p_right = integrate.simpson(Pm[Nx//2:], X[Nx//2:])
#
#     measure = 0 if p_left + p_right == 0 else np.abs(p_left - p_right) / (p_left + p_right)
#
#     # Orientation
#     if p_right > p_left:  # P is on the right
#         orientation = 1
#     elif p_right < p_left:  # P is on the left
#         orientation = 2
#     else:  # undetermined
#         orientation = 0
#
#     return measure, orientation, orientation_marker(orientation)

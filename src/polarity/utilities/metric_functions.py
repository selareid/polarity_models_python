import numpy as np
from scipy import integrate
from polarity.utilities.figure_helper import convert_output_to_pandas
from polarity.model_enums import MODELS


# Polarity Measure -----------------------------------------------------------------------------------------------------
# A,P just on membrane
def polarity_measure(X, Y, model: MODELS):
    species = ["A", "P"] if model == MODELS.GOEHRING else ["J", "M", "A", "P"]
    df = convert_output_to_pandas(X, Y, species = species)
    measure, _, _ = polarity_get_all(X, df.A, df.P)
    return measure


# determine the orientation of polarity
# 0 - undetermined
# 1 - anterior on left
# 2 - anterior on right
def polarity_orientation(X, Am, Pm):
    _, orientation, _ = polarity_get_all(X, Am, Pm)
    return orientation


def orientation_marker(orientation_code):
    return ['o', '<', '>'][orientation_code]


# returns (measure, orientation, marker)
def polarity_get_all(X, Am, Pm):
    Nx = len(X)
    a_left = integrate.simpson(Am[:Nx//2], x = X[:Nx//2])
    a_right = integrate.simpson(Am[Nx//2:], x = X[Nx//2:])
    b_left = integrate.simpson(Pm[:Nx//2], x = X[:Nx//2])
    b_right = integrate.simpson(Pm[Nx//2:], x = X[Nx//2:])

    measure = 0 
    if (((a_left + a_right)*(b_left + b_right)) != 0): 
        #measure = np.abs(a_left - a_right) * np.abs(b_left - b_right) / ((a_left + a_right)*(b_left + b_right))
        measure = -(a_left - a_right) * (b_left - b_right) / ((a_left + a_right)*(b_left + b_right))

    # Orientation
    if a_left > a_right and b_right > b_left:  # A is on the left, B is on the right
        orientation = 1
    elif a_left < a_right and b_right < b_left:  # A is on the right, B is on the left
        orientation = 2
    else:  # undetermined
        orientation = 0

    return measure, orientation, orientation_marker(orientation)



# Find the interface (crossover point) between the basal (YB) and apical (YA) species ---------------------------------------
def find_interface(X, Y, model: MODELS):

    # Get appropriate species for the model
    species = ["J", "M", "A", "P"] if model != MODELS.GOEHRING else ["A", "P"]
    apical_species = "J" if model != MODELS.GOEHRING else "A"
    df = convert_output_to_pandas(X, Y, species = species)
    YB = df.P # Basal species (same between models)
    YA = df[apical_species] # Apical species
    
    # Find the indices where YB < YA
    indices_BltA = [i for i, B, A in zip( range(len(X)), YB, YA ) if B < A] # index where B < A
    
    # If it never crosses, return 1
    if (len(indices_BltA) == 0):
        return 1
    
    # Otherwise linearly interpolate to find the first the crossing point
    j = np.min(indices_BltA)
    i = j-1
    if (j == 0):
        return 0 # crossing point is at the left boundary of the domain
    
    slope_YB = (YB[j]-YB[i])/(X[j]-X[i])
    slope_YA = (YA[j]-YA[i])/(X[j]-X[i])
    crossing_point = (YB[i] - YA[i])/(slope_YA - slope_YB) + X[i]
    return  crossing_point

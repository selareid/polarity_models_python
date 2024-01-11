from enum import Enum
import numpy as np
from scipy import integrate

from . import goehring, tostevin


class MODELS(Enum):
    TOSTEVIN = 0
    GOEHRING = 1


def model_to_module(model: MODELS):
    match model:
        case MODELS.TOSTEVIN:
            return tostevin
        case MODELS.GOEHRING:
            return goehring
        case _:
            raise ValueError(f"Unexpected model value: {model}")


# Analysis Metric
# A,P just on membrane
def polarity_measure(X, Am, Pm, Nx):
    a_left = integrate.simpson(Am[:Nx//2], X[:Nx//2])
    a_right = integrate.simpson(Am[Nx//2:], X[Nx//2:])
    p_left = integrate.simpson(Pm[:Nx//2], X[:Nx//2])
    p_right = integrate.simpson(Pm[Nx//2:], X[Nx//2:])

    return 0 if ((a_left + a_right)*(p_left + p_right)) == 0 else np.abs(a_left - a_right) * np.abs(p_left - p_right) / ((a_left + a_right)*(p_left + p_right))


# determine the orientation of polarity
# 0 - undetermined
# 1 - anterior on left
# 2 - anterior on right
def polarity_orientation(X, Am, Pm, Nx):
    a_left = integrate.simpson(Am[:Nx // 2], X[:Nx // 2])
    a_right = integrate.simpson(Am[Nx // 2:], X[Nx // 2:])
    p_left = integrate.simpson(Pm[:Nx // 2], X[:Nx // 2])
    p_right = integrate.simpson(Pm[Nx // 2:], X[Nx // 2:])

    if a_left > a_right and p_right > p_left:  # A is on the left, P is on the right
        return 1
    elif a_left < a_right and p_right < p_left:  # A is on the right, P is on the left
        return 2
    else:  # undetermined
        return 0


def orientation_marker(orientation_code):
    return ['o', '<', '>'][orientation_code]
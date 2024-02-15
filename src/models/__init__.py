from enum import Enum
from . import goehring, tostevin, par3addition, crumbs


class MODELS(Enum):
    TOSTEVIN = 0
    GOEHRING = 1
    PAR3ADD = 2
    CRUMBS = 3


def model_to_module(model: MODELS):
    match model:
        case MODELS.TOSTEVIN:
            return tostevin
        case MODELS.GOEHRING:
            return goehring
        case MODELS.PAR3ADD:
            return par3addition
        case MODELS.CRUMBS:
            return crumbs
        case _:
            raise ValueError(f"Unexpected model value: {model}")
from enum import Enum
from . import goehring, par3addition


class MODELS(Enum):
    GOEHRING = 0
    PAR3ADD = 1


def model_to_module(model: MODELS):
    match model:
        case MODELS.GOEHRING:
            return goehring
        case MODELS.PAR3ADD:
            return par3addition
        case _:
            raise ValueError(f"Unexpected model value: {model}")


def model_to_string(model: MODELS):
    match model:
        case MODELS.GOEHRING:
            return "goehring"
        case MODELS.PAR3ADD:
            return "par3addition"
        case _:
            raise ValueError(f"Unexpected model value: {model}")
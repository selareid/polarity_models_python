from enum import Enum
from polarity.models import goehring, par3addition, par3nd

class MODELS(Enum):
    GOEHRING = 0
    PAR3ADD = 1
    PAR3ND = 2


def model_to_module(model: MODELS):
    match model:
        case MODELS.GOEHRING:
            return goehring
        case MODELS.PAR3ADD:
            return par3addition
        case MODELS.PAR3ND:
            return par3nd
        case _:
            raise ValueError(f"Unexpected model value: {model}")


def model_to_string(model: MODELS):
    match model:
        case MODELS.GOEHRING:
            return "goehring"
        case MODELS.PAR3ADD:
            return "par3addition"
        case MODELS.PAR3ND:
            return "par3nd"
        case _:
            raise ValueError(f"Unexpected model value: {model}")
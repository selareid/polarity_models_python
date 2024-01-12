from enum import Enum
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
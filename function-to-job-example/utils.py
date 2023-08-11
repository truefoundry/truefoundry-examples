import inspect
from typing import List, Callable

from servicefoundry import Param
from pydantic import BaseModel


class GeneratedParams(BaseModel):
    params: List[Param]
    command_argument: str


def generate_params(func: Callable) -> GeneratedParams:
    params = []
    command_argument = ""

    sig = inspect.signature(func)

    for name, param in sig.parameters.items():
        params.append(
            Param(
                name=name,
                default=param.default
                if param.default != inspect.Parameter.empty
                else None,
            )
        )

    command_argument = " ".join(
        f"--{param.name} {{{{{param.name}}}}}" for param in params
    )

    return GeneratedParams(params=params, command_argument=command_argument)


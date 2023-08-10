import inspect
from typing import Dict, List

from servicefoundry import Param


def generate_params(func) -> List[Dict[str, str]]:
    sig = inspect.signature(func)
    args = []

    for name, param in sig.parameters.items():
        arg_info = {
            "name": name,
            "default": param.default
            if param.default != inspect.Parameter.empty
            else None,
        }
        args.append(arg_info)

    params = [
        Param(
            name=arg_info["name"],
            default=arg_info["default"],
        )
        for arg_info in args
    ]

    return params

import inspect
import argparse
import logging
from typing import List
from servicefoundry import Param
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

class Params(BaseModel):
    name: str
    default: str

class GeneratedParams(BaseModel):
    params: List[Params]
    command_argument: str

def generate_params(callable) -> GeneratedParams:
    params_list = []
    command_argument = ""

    sig = inspect.signature(callable) 

    for name, param in sig.parameters.items():
        param_info = {
            "name": name,
            "default": param.default
            if param.default != inspect.Parameter.empty
            else None,
        }
        params_list.append(param_info)

    params = [
        Param(
            name=param_info["name"],
            default=param_info["default"],
        )
        for param_info in params_list
    ]

    command_argument = " ".join(f"--{param_info['name']} {{{{{param_info['name']}}}}}" for param_info in params_list)
  
    return GeneratedParams(params=params, command_argument=command_argument)
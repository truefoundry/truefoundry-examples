import typer
from typing_extensions import Annotated
import numpy as np

def normal(loc: Annotated[float, typer.Option()] = 1, scale: Annotated[float, typer.Option()] = 2):
    return np.random.normal(loc=loc, scale=scale).tolist()
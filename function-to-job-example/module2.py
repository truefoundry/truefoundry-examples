import typer
from typing_extensions import Annotated
import numpy as np


def uniform(
    low: Annotated[float, typer.Option()] = 1,
    high: Annotated[float, typer.Option()] = 1,
):
    return np.random.uniform(low=low, high=high)


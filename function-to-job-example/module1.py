import typer
from typing_extensions import Annotated
import numpy as np


def normal(
    loc: Annotated[float, typer.Option()] = 1,
    scale: Annotated[float, typer.Option()] = 2,
):
    print(f"Executing normal with loc {loc} scale {scale}")
    return np.random.normal(loc=loc, scale=scale)


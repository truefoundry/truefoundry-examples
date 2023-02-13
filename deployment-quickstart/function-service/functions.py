from typing import List
import numpy as np


def multiply(a: float, b: float):
  return a * b 


def normal(loc: float, scale: float, size: List[int]):
    return np.random.normal(loc=loc, scale=scale, size=size).tolist()


def uniform(low: float, high: float, size: List[int]):
    return np.random.uniform(low=low, high=high, size=size).tolist()

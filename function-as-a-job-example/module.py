import numpy as np

def normal(loc: float, scale: float):
    return np.random.normal(loc=loc, scale=scale).tolist()


def uniform(low: float, high: float):
    return np.random.uniform(low=low, high=high).tolist()
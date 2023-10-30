import numpy as np


def vec(*vals):
    """
    Create a numpy column vector from the given args
    >>> vec(1, 2, 3)
    >>> np.array([
        [1],
        [2],
        [3]
    ])
    """
    return np.vstack([*vals])
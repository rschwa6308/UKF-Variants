import numpy as np
import jax.numpy as jnp


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


def lerp(a, b, t):
    return a * (1 - t) + b * t


def interval_overlap(interval1, interval2):
    low1, high1 = interval1
    low2, high2 = interval2
    overlap = min(high1, high2) - max(low1, low2)

    return max(0, overlap)

assert(interval_overlap((0, 10), (2, 8)) == 6)
assert(interval_overlap((0, 10), (5, 15)) == 5)
assert(interval_overlap((0, 10), (10, 20)) == 0)
assert(interval_overlap((0, 10), (11, 20)) == 0)


def wrap2pi(angles):
    """
    Wraps an array of angles into [-pi, pi].

    (JIT-compatible implementation)
    """
    
    angles_wrapped = jnp.mod(angles + jnp.pi, 2*jnp.pi) - jnp.pi
    return angles_wrapped


def cartesian_product(arrays):
    "Compute the explicit (dense) cartesian product of given arrays"
    la = len(arrays)

    if la == 1:
        dtype = arrays[0].dtype
    else:
        dtype = np.result_type(*arrays)
    
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr

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


def interval_overlap(interval1, interval2):
    low1, high1 = interval1
    low2, high2 = interval2
    overlap = min(high1, high2) - max(low1, low2)

    return max(0, overlap)


def lerp(a, b, t):
    return a * (1 - t) + b * t


def query_pdf(domain, pdf_values, x, interp=True):
    domain_step = domain[1] - domain[0]
    index = (x - domain[0]) / domain_step

    index -= 0.5        # PDF samples at bin midpoint!

    if not interp:
        return pdf_values[int(index)]

    else:
        index_low, index_high = int(np.floor(index)), int(np.ceil(index))
        index_low = max(0, index_low)
        index_high = min(len(pdf_values)-1, index_high)
        t = index - index_low
        return lerp(pdf_values[index_low], pdf_values[index_high], t)



assert(interval_overlap((0, 10), (2, 8)) == 6)
assert(interval_overlap((0, 10), (5, 15)) == 5)
assert(interval_overlap((0, 10), (10, 20)) == 0)
assert(interval_overlap((0, 10), (11, 20)) == 0)
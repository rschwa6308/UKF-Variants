import numpy as np
from typing import Callable, Tuple

from probability_distributions import ProbabilityDistribution, GaussianDistribution


"""
An 'uncertainty transform' models the affect of a function applied to a value that is
known with some uncertainty. Such a transform generally takes as input
 - the function being applied to the domain $f(X): R^n -> R^m$
 - the probability distribution representing our prior on the value $P(X): R^n -> R$
and returns
 - the approximate posterior belief on the value: $P(Y|X): R^n -> R$
 - potentially some additional information on the joint distribution $P(X,Y)$
"""


from sigma_points import SigmaPointSelector


def unscented_transform(func: Callable, mean_x, cov_xx, sigma_point_selector: SigmaPointSelector) -> Tuple:
    """
    Given a function f(X) = Y together with the mean and covariance on X,
    apply the unscented transform to obtain the mean on Y, the covariance on Y, and the cross-covariance on X and Y
    """

    # select sigma points in accordance with prior
    sigma_points, weights_mean, weights_cov = sigma_point_selector.select_sigma_points(mean_x, cov_xx)

    # pass sigma points through the function
    sigma_points_transformed = []
    for i in range(sigma_points.shape[1]):
        point = sigma_points[:,[i]]
        x_prime = func(point)
        sigma_points_transformed.append(x_prime)
    
    sigma_points_transformed = np.hstack(sigma_points_transformed)

    # fit a gaussian to weighted transformed sigma points
    mean_y = np.sum(weights_mean * sigma_points_transformed, axis=1, keepdims=True)
    cov_yy = weights_cov * (sigma_points_transformed - mean_y) @ (sigma_points_transformed - mean_y).T
    cov_xy = weights_cov * (sigma_points - mean_x) @ (sigma_points_transformed - mean_y).T

    return mean_y, cov_yy, cov_xy

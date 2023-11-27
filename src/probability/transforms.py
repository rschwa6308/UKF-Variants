import numpy as np
from typing import Callable, Tuple
from helpers import interval_overlap

from probability.distributions import ProbabilityDistribution, GaussianDistribution, HistogramDistribution


"""
An 'uncertainty transform' models the affect of a function applied to a value that is
known with some uncertainty. Such a transform generally takes as input
 - the function being applied to the domain $f(X): R^n -> R^m$
 - the probability distribution representing our prior on the value $P(X): R^n -> R$
and returns
 - the approximate posterior belief on the value: $P(Y|X): R^n -> R$
 - potentially some additional information on the joint distribution $P(X,Y)$
"""


from probability.sigma_points import SigmaPointSelector


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




def add_mass_to_pdf_in_interval(domain, pdf, low, high, mass):
    # assert(high > low)

    # convert to index space
    # domain_step = (domain[-1] - domain[0]) / (len(domain) - 1)
    domain_step = domain[1] - domain[0]
    low_idx = (low - domain[0]) / domain_step
    high_idx = (high - domain[0]) / domain_step

    width = high_idx - low_idx
    
    if width == 0:
        pdf[int(low_idx)] += mass
        return
    
    for x in range(int(np.floor(low_idx)), int(np.ceil(high_idx))):
        overlap = interval_overlap((x, x+1), (low_idx, high_idx))
        pdf[x] += mass * overlap / width




def histogram_transform(func: Callable, pdf: HistogramDistribution):
    if pdf.dim > 1:
        raise NotImplementedError("Histogram transform for dim > 1 not yet supported.")

    # apply function to the entire domain
    func_values = func(pdf.domain)
    func_values_min, func_values_max = (np.min(func_values), np.max(func_values))

    # define the output domain to have sample density (approximately) equal to input domain
    input_domain_volume = pdf.domain[-1] - pdf.domain[0]
    output_domain_volume = func_values_max - func_values_min
    output_domain = np.linspace(
        func_values_min, func_values_max,
        len(pdf.domain) * (output_domain_volume / input_domain_volume)
    )

    output_pdf = np.zeros_like(output_domain)

    # iterate over bins - TODO: vectorize
    for i in range(len(pdf.domain) - 1):
        mass = pdf[i] * (pdf.domain[i+1] - pdf.domain[i])   # mass in source bin [i, i+1]
        image = (func_values[i], func_values[i+1])          # mass transport destination
    
        # Distribute mass across output bins intersecting the destination region

        # The image is a polytope with 2*input_dim vertices,
        # (topologically a hypercube).
        # It will be degenerate if output_dim < input_dim
        # - in 1D: interval
        # - in 2D: quadrilateral
        # - in 3D: octahedron
        # - in 4D: 16-cell

        
        # add_mass_to_pdf_in_interval(domain, output_pdf, image[0], image[1], mass)

        # use a rasterization scheme
        image_bounding_box = TODO

    return output_pdf



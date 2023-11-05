import numpy as np
from typing import Tuple

from matrix_square_root import *



class SigmaPointSelector:
    def select_sigma_points(self, mean, cov) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        "Select sigma points and weights according to the given distribution. Returns (points, weights_mean, weights_cov)."
        raise NotImplementedError()



class StandardSigmaPointSelector(SigmaPointSelector):
    def __init__(self, alpha=0.1, beta=2.0, kappa=0.0, mat_sqrt_alg=mat_sqrt_cholesky):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.mat_sqrt_alg = mat_sqrt_alg
    
    def select_sigma_points(self, mean, cov) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = cov.shape[0]    # dimensionality

        # compute covariance sqrt
        cov_sqrt = self.mat_sqrt_alg(cov)

        # compute shell scaling factor
        lam = self.alpha**2 * (n + self.kappa) - n
        shell_scale = np.sqrt(n + lam)

        sigma_points = []
        weights_mean = []
        weights_cov = []

        # include the mean
        sigma_points.append(mean)
        weights_mean.append(lam / (n + lam))
        weights_cov.append(lam / (n + lam) + (1 - self.alpha**2 + self.beta))
        
        # include 2n points in a single shell around the mean
        for i in range(n):
            vec = shell_scale * cov_sqrt[:,i].reshape(-1, 1)
            
            sigma_points.append(mean + vec)
            sigma_points.append(mean - vec)

            weights_mean.append(1 / (2*(n + lam)))
            weights_mean.append(1 / (2*(n + lam)))

            weights_cov.append(1 / (2*(n + lam)))
            weights_cov.append(1 / (2*(n + lam)))

        return (np.hstack(sigma_points), np.array(weights_mean), np.array(weights_cov))

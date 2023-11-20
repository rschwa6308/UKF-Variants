import numpy as np

from helpers import lerp

class ProbabilityDistribution:
    def __init__(self, dim: int):
        self.dim = dim

    def pdf(self, x):
        pass


class GaussianDistribution(ProbabilityDistribution):
    def __init__(self, mean, covariance):
        dim = covariance.shape[0]
        super().__init__(dim)

        self.mean = mean
        self.covariance = covariance
    
    def pdf(self, x):
        # TODO: verify math
        cov_det = np.linalg.det(self.covariance)
        cov_inv = np.linalg.inv(self.covariance)
        
        eta = 1 / np.sqrt((2*np.pi)**self.dim * cov_det)
        vals = eta * np.exp(-0.5 * (x - self.mean).T @ cov_inv @ (x - self.mean))
        return vals
    
    def __repr__(self):
        return f"GaussianDistribution(mean={self.mean}, covariance={self.covariance})"


class HistogramDistribution(ProbabilityDistribution):
    """
    A direct numerical representation of a PDF, resembling a multi-dimensional histogram. Bins are defined as follows:

    `values[i,...,k]` represents the average probability density in the
    rectangular region between `domain[i,...,k]` and `domain[i+1,...,k+1]`

    The total probability mass of a bin is the product of its density value and its volume. This means `np.sum(values * (domain[1:] - domain[:1])) == 1.0` (in 1D).

    For example,
    ```
    >>> domain = np.array([ 0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    >>> values = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
    >>> HistogramDistribution(domain, values)
    ```
    yields the uniform distribution on (0, 50).
    """

    def __init__(self, domain, values):
        assert(all(domain_dim == values_dim + 1 for domain_dim, values_dim in zip(domain.shape, values.shape)))
        assert(np.sum(values) == 1.0)

        self.domain = domain
        self.values = values

        self.step = domain[1] - domain[0]
        self.domain_bounds = (domain[0], domain[-1]+self.step)
    
    def pdf(self, x, interp=True):
        index = (x - self.domain[0]) / self.step

        index -= 0.5        # PDF samples at bin midpoint!

        if not interp:
            return self.values[int(index)]

        index_low, index_high = int(np.floor(index)), int(np.ceil(index))

        index_low = np.max(0, index_low)
        index_high = np.min(len(self.values)-1, index_high)
        
        t = index - index_low
        return lerp(self.values[index_low], self.values[index_high], t)

    def __repr__(self):
        return f"HistogramDistribution(bins=[({self.domain[0]:.3f}, {self.domain[1]:.3f}), ..., ({self.domain[-2]:.3f}, {self.domain[-1]:.3f})])"




class ParticleDistribution(ProbabilityDistribution):
    pass    # TODO



def convolve_distributions(pdf1: ProbabilityDistribution, pdf2: ProbabilityDistribution):
    pass        # TODO

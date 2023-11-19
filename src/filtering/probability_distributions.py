import numpy as np

class ProbabilityDistribution:
    def pdf(self, x):
        pass


class GaussianDistribution(ProbabilityDistribution):
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
    
    def pdf(self, x):
        # TODO: verify math
        dim = self.covariance.shape[0]

        cov_det = np.linalg.det(self.covariance)
        cov_inv = np.linalg.inv(self.covariance)
        
        eta = 1 / np.sqrt((2*np.pi)**dim * cov_det)
        vals = eta * np.exp(-0.5 * (x - self.mean).T @ cov_inv @ (x - self.mean))
        return vals


class HistogramDistribution(ProbabilityDistribution):
    pass    # TODO


class ParticleDistribution(ProbabilityDistribution):
    pass    # TODO


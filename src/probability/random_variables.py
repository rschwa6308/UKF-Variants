import numpy as np
from typing import List, Tuple

from probability.distributions import ProbabilityDistribution, GaussianDistribution, convolve_distributions



class RandomVariable:
    def __init__(self, support: List[Tuple[float, float]], pdf: ProbabilityDistribution):
        self.support = np.array(support)
        self.dim = len(support)
        self.pdf = pdf

        if isinstance(pdf, GaussianDistribution):
            if not np.all(np.isinf(self.support)):
                raise ValueError("Random variable with Gaussian PDF must have infinite support. Specify support=[(-np.inf, +np.inf), ...].")
    
    def __repr__(self):
        return f"RandomVariable(support={self.support}, pdf={self.pdf})"

    def __add__(self, other: "RandomVariable"):
        support_sum = self.support + other.support
        pdf_convolution = convolve_distributions(self.pdf, other.pdf)
        return RandomVariable(support_sum, pdf_convolution)

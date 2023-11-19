class ProbabilityDistribution:
    pass


class GaussianDistribution(ProbabilityDistribution):
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance


class HistogramDistribution(ProbabilityDistribution):
    pass    # TODO


class ParticleDistribution(ProbabilityDistribution):
    pass    # TODO


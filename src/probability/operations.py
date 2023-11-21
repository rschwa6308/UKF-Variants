import numpy as np

from probability.distributions import *



def convolve_gaussian_with_gaussian(pdf1: GaussianDistribution, pdf2: GaussianDistribution):
    return GaussianDistribution(
        pdf1.mean + pdf2.mean,
        pdf1.covariance + pdf2.covariance
    )

def convolve_histogram_with_histogram(pdf1: HistogramDistribution, pdf2: HistogramDistribution):
    # TODO
    np.convolve()


convolution_routines = {
    ("GaussianDistribution", "GaussianDistribution"):  convolve_gaussian_with_gaussian,
    # ("GaussianDistribution", "HistogramDistribution"): convolve_histogram_with_gaussian,
    # ("HistogramDistribution", "GaussianDistribution"): convolve_histogram_with_gaussian,
    ("HistogramDistribution", "HistogramDistribution"): convolve_histogram_with_histogram
}


def convolve_distributions(pdf1: ProbabilityDistribution, pdf2: ProbabilityDistribution) -> ProbabilityDistribution:
    routine = convolution_routines.get((pdf1.type, pdf2.type), None)

    if routine is None:
        raise RuntimeError(f"Convolution is not yet supported between {type(pdf1)} and {type(pdf2)}")

    return routine(pdf1, pdf2)
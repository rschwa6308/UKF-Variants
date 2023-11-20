import numpy as np

from helpers import lerp, cartesian_product


class ProbabilityDistribution:
    def __init__(self, dim: int):
        self.dim = dim

    def pdf(self, x):
        pass

    def get_mean(self):
        pass

    def get_covariance(self):
        pass


class GaussianDistribution(ProbabilityDistribution):
    def __init__(self, mean, covariance):
        dim = covariance.shape[0]
        super().__init__(dim)

        self.mean = mean
        self.covariance = covariance
    
    def pdf(self, x):
        assert(x.shape[-1] == self.dim)

        x_flat = x.reshape(-1, self.dim)

        # TODO: verify math
        cov_det = np.linalg.det(self.covariance)
        cov_inv = np.linalg.inv(self.covariance)
        
        eta = 1 / np.sqrt((2*np.pi)**self.dim * cov_det)
        print((x - self.mean).shape)
        vals = eta * np.exp(-0.5 * np.sum((x_flat - self.mean) @ cov_inv * (x_flat - self.mean), axis=1))

        vals = vals.reshape(x.shape[:-1])
        return vals
    
    def __repr__(self):
        return f"GaussianDistribution(mean={self.mean}, covariance={self.covariance})"
    
    def get_mean(self):
        return self.mean
    
    def get_covariance(self):
        return self.covariance



class HistogramDistribution(ProbabilityDistribution):
    """
    A direct numerical representation of a PDF, formulated as a D-dimensional histogram over a rectangular region with uniform bins.

     - `domain_bounds`: a list of D 2-tuples representing the lower and upper bounds on each dimension of the domain
     - `bin_counts`: an array of integers representing the number of bins to discretize each dimension into
     
    The `domain` itself is constructed internally as the cartesian product of `[np.linspace(*domain_bounds[i], bin_counts[i]+1) for i in range(D)]`

     - `values[i,...,k]` represents the total probability mass in the rectangular region between
       `domain[i,...,k]` and `domain[i+1,...,k+1]`, and thus has shape one less than `domain` along each dimension:
       `(domain.shape[0]-1, ..., domain.shape[-1]-1)`


    Example (1D): the uniform distribution on (0, 50):
    ```
    >>> domain_bounds = [(0.0, 5.0)]
    >>> bin_counts = [5]
    >>> values =  np.array([0.2,  0.2,  0.2,  0.2,  0.2])
    >>> HistogramDistribution(domain_bounds, bin_counts, values)
    ```

    Example (2D): the uniform distribution on (-1, +1) x (-5, 5):
    ```
    >>> domain_bounds = [(-1.0, +1.0), (-5.0, 5.0)]
    >>> bin_counts = [200, 1000]      # yields square bins
    >>> values = np.ones((200, 1000))
    >>> values /= np.sum(values)
    >>> HistogramDistribution(domain_bounds, bin_counts, values)
    ```
    """

    def __init__(self, domain_bounds, bin_counts, values):
        assert(len(domain_bounds) == len(bin_counts))
        assert(all(domain_dim == values_dim for domain_dim, values_dim in zip(bin_counts, values.shape)))
        assert(np.isclose(np.sum(values), 1.0))

        super().__init__(len(domain_bounds))

        self.domain_bounds = np.array(domain_bounds)
        self.bin_counts = np.array(bin_counts)

        self.domain = cartesian_product([
            np.linspace(*domain_bounds[i], bin_counts[i]+1)
            for i in range(self.dim)
        ])

        self.steps = (self.domain_bounds[:,1] - self.domain_bounds[:,0]) / self.bin_counts

        self.bin_lowers = self.domain[(np.s_[:-1],) * self.dim]
        self.bin_uppers = self.domain[(np.s_[1:],) * self.dim]
        self.bin_midpoints = self.bin_lowers + self.steps/2

        self.values = values

    def pdf(self, x, interp=False):
        x_flat = x.reshape(-1, self.dim)

        # mask out query points that are outside the domain - they will be assigned value 0
        mask = np.all((x_flat >= self.domain_bounds[:,0]) & (x_flat < self.domain_bounds[:,1]), axis=1)

        index = (x_flat - self.domain_bounds[:,0]) / self.steps
        # index -= 0.5        # PDF samples at bin midpoint!

        if not interp:
            # find the bin the query point belongs to
            index_bin = np.floor(index).astype(int)

            # return 0 if point is outside domain
            vals = np.zeros((x_flat.shape[0]))
            vals[mask] = np.take(self.values,
                np.ravel_multi_index(index_bin[mask].T, self.values.shape)
            )
        
        else:
            raise NotImplementedError()
            # index_low, index_high = np.floor(index).astype(int), np.ceil(index).astype(int)

            # index_low = np.clip(index_low, 0, None)
            # index_high = np.clip(index_high, None, self.bin_counts-1)
            # print(index)
            # print(index_low)
            # print(index_high)

            # t = index - index_low
            # print(self.values.shape)
            # values_low = np.take(self.values,
            #     np.ravel_multi_index(index_low.T, self.values.shape)
            # )
            # values_high = np.take(self.values,
            #     np.ravel_multi_index(index_high.T, self.values.shape)
            # )

            # print(values_low)
            # print(values_high)
            # print(t)

            # vals = lerp(values_low, values_high, t.reshape(-1, 1))

        vals = vals.reshape(x.shape[:-1])
        return vals

    def __repr__(self):
        return f"HistogramDistribution(domain_bounds=[{', '.join(map(str, self.domain_bounds))}], bin_counts={self.bin_counts})"

    def get_mean(self):
        values_flat = self.values.reshape(-1, 1)
        bin_midpoints_flat = self.bin_midpoints.reshape(-1, self.dim)
        return np.sum(values_flat * bin_midpoints_flat)

    def get_covariance(self):
        mean = self.get_mean()
        values_flat = self.values.reshape(-1, 1)
        bin_midpoints_flat = self.bin_midpoints.reshape(-1, self.dim)
        return (bin_midpoints_flat - mean).T @ (values_flat * (bin_midpoints_flat - mean))




class ParticleDistribution(ProbabilityDistribution):
    pass    # TODO



def convolve_distributions(pdf1: ProbabilityDistribution, pdf2: ProbabilityDistribution):
    pass        # TODO

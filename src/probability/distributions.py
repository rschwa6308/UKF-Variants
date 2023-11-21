from typing import List
import numpy as np
from matplotlib import pyplot as plt

from helpers import lerp, cartesian_product, format_matrix
from probability.visualization import plot_covariance_ellipse, plot_pdf_values
from mpl_toolkits.axes_grid1 import make_axes_locatable



class ProbabilityDistribution:
    type = "GENERIC"

    def __init__(self, dim: int):
        self.dim = dim

    def pdf(self, x) -> float:
        pass

    def get_mean(self):
        pass

    def get_covariance(self):
        pass

    def plot(self, ax: plt.Axes=None, label="DEFAULT", cmap="Blues", contours_filled=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        fig = ax.figure
        
        if label == "DEFAULT":
            label = str(self)
        
        mean, cov = self.get_mean(), self.get_covariance()

        if self.dim == 1:
            "plot PDF values numerically"
    
            # center axis at the mean with padding scaled wrt largest std value
            ax_width = 5 * np.sqrt(np.max(np.diag(cov)))
            x_low, x_high = mean[0] - ax_width, mean[0] + ax_width

            samples = np.linspace(x_low, x_high, 1000)    # large number of sample points
            pdf_values = self.pdf(samples.reshape(-1, 1)).flatten()
            plot_pdf_values(ax, samples, pdf_values, label=label)
            
            # set axis limits
            ax.set_xlim(samples[0], samples[-1])
            ax.set_ylim(0.0, np.max(pdf_values) * 1.2)      # include some vertical padding
        
        if self.dim == 2:
            "plot PDF values numerically using contourf"

            # center axis at the mean with padding scaled wrt largest std value
            ax_width = 3 * np.sqrt(np.max(np.diag(cov)))
            x_low, x_high = mean[0] - ax_width, mean[0] + ax_width
            y_low, y_high = mean[1] - ax_width, mean[1] + ax_width

            ax.axis("equal")

            samples_x = np.linspace(x_low, x_high, 100)    # large number of sample points
            samples_y = np.linspace(y_low, y_high, 100)    # large number of sample points
            samples = cartesian_product([samples_x, samples_y])
            pdf_values = self.pdf(samples)

            if contours_filled:
                cs = ax.contourf(samples_x, samples_y, pdf_values.T, cmap=cmap, **kwargs)
                # cs = ax.pcolormesh(samples_x, samples_y, pdf_values.T, cmap=cmap, **kwargs)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(cs, cax=cax)
            else:
                ax.contour(samples_x, samples_y, pdf_values.T, cmap=cmap, **kwargs)
            
            # set axis limits
            ax.set_xlim(x_low, x_high)
            ax.set_ylim(y_low, y_high)

        if label is not None and self.dim != 2:
            ax.legend()



class ParametricDistribution(ProbabilityDistribution):
    pass


class GaussianDistribution(ParametricDistribution):
    type = "GaussianDistribution"

    def __init__(self, mean, covariance):
        dim = covariance.shape[0]
        super().__init__(dim)

        self.mean = mean
        self.covariance = covariance

    def pdf(self, x):
        x_flat = x.reshape(-1, self.dim)

        cov_det = np.linalg.det(self.covariance)
        cov_inv = np.linalg.inv(self.covariance)
        
        eta = 1 / np.sqrt((2*np.pi)**self.dim * cov_det)
        vals = eta * np.exp(-0.5 * np.sum((x_flat - self.mean) @ cov_inv * (x_flat - self.mean), axis=1))

        vals = vals.reshape(x.shape[:-1])
        return vals
    
    def __repr__(self):
        return f"GaussianDistribution(mean={self.mean}, covariance={self.covariance})"
    
    def get_mean(self):
        return self.mean
    
    def get_covariance(self):
        return self.covariance

    
    def plot(self, ax: plt.Axes=None, label="DEFAULT", mode_2D="contour_plot", **kwargs):
        if self.dim != 2 or mode_2D != "ellipse":
            super().plot(ax, label=label, **kwargs)
            return

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
                
        if label == "DEFAULT":
            mean_str = format_matrix([self.mean])
            cov_str = "\n    ".join(format_matrix(self.covariance).split("\n"))
            label = f"μ = {mean_str}\nΣ = {cov_str}"
        
        # center axis at the mean with padding scaled wrt largest std value
        ax_width = 3 * np.sqrt(np.max(np.diag(self.covariance)))
        x_low, x_high = self.mean[0] - ax_width, self.mean[0] + ax_width
        y_low, y_high = self.mean[1] - ax_width, self.mean[1] + ax_width

        ax.axis("equal")

        "draw covariance ellipse on domain"
        plot_covariance_ellipse(ax, self.mean, self.covariance, label=label)
        ax.plot(*self.mean, marker=".", color="black")     # point at mean

        # set axis limits
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)

        if label is not None:
            ax.legend()



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

    type = "HistogramDistribution"

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
        return np.sum(values_flat * bin_midpoints_flat, axis=0)

    def get_covariance(self):
        mean = self.get_mean()
        values_flat = self.values.reshape(-1, 1)
        bin_midpoints_flat = self.bin_midpoints.reshape(-1, self.dim)
        return (bin_midpoints_flat - mean).T @ (values_flat * (bin_midpoints_flat - mean))




class MixtureDistribution(ProbabilityDistribution):
    def __init__(self, components: List[ProbabilityDistribution], weights):
        assert(all(c.dim == components[0].dim for c in components))
        assert(np.isclose(np.sum(weights), 1.0))
        assert(len(components) == len(weights))

        super().__init__(components[0].dim)

        self.components = components
        self.weights = np.array(weights)
    
    def pdf(self, x):
        assert(x.shape[-1] == self.dim)
        x_flat = x.reshape(-1, self.dim)

        component_values = np.array([c.pdf(x_flat) for c in self.components])
        vals = np.sum(self.weights.reshape(-1, 1) * component_values, axis=0)

        vals = vals.reshape(x.shape[:-1])
        return vals
    
    def get_mean(self):
        component_means = np.array([c.get_mean() for c in self.components])
        mean = np.sum(self.weights.reshape(-1, 1) * component_means, axis=0)
        return mean
    
    def get_covariance(self):
        # General derivation can be found here:
        # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        component_means = np.array([c.get_mean() for c in self.components])
        component_covs = np.array([c.get_covariance() for c in self.components])

        weights = self.weights.reshape(-1,1)
        mean = np.sum(weights * component_means, axis=0)

        cov = np.sum(weights.reshape(-1,1,1) * component_covs, axis=0) \
            + component_means.T @ (weights * component_means) \
            - mean.reshape(self.dim, 1) @ mean.reshape(1, self.dim)
        return cov




class ParticleDistribution(ProbabilityDistribution):
    type = "ParticleDistribution"
    pass    # TODO


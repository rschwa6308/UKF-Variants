# 16-833 Robot Localization and Mapping - Final Project Proposal

Idea: Extensions to the Unscented Kalman Filter


## Motivation

The EKF extends the Kalman Filter by approximating the system dynamics (and measurement model) via 1st-order taylor expansion at the current state estimate. The EKF is non-optimal, in the sense that the resulting gaussian is (in general) not the best-fitting gaussian for the underlying posterior. 

One way to compute this best-fitting gaussian (or at least a very close approxiamtion to it) would be to use a histogram-based approach, where we discretize the state space into a set of bins, pass each bin through our non-linear dynamics, and then fit a gaussian to the resulting histogram-distribution. However, this approach is impractical (and it defeats the purpose of using a parametrized belief in the first place).

The UKF provides a better approach, which is more akin to a particle filter. A set of so-called "sigma points" are sampled from the $d$-dimensional state space in accordance with the prior:
 - 1 point at the mean
 - $2d$ points distributed symmetrically about the mean, at a fixed distance (mahalanobis distance)

We pass each of these points through our non-linear dynamics, and then fit a gaussian to the resulting particle-distribution, taking care to weight the points relative to their probability density in the prior. Impressively, the resulting estimate captures the posterior mean and covariance accurately to the 3rd order (for *any* non-linearity).

The UKF results in a better-fitting gaussian than the EKF. Intuitively, this is in part because the UKF samples more of the state space and thus consults more of the dynamics model. In a sense, the UKF is an interpolation between single-sample methods (like the EKF) and many-sample methods (like the histogram-based approach). 

This perspective begs the question: Are there other interpolations along this axis that may yield even better approximations than the UKF? Are there other approaches orthogonal to this axis? Can we characterize the types of systems in which these alternatives to the UKF perform better than the original?



## Possible Directions

Analytic Questions
 - Can the EKF be made to perform arbitrarily bad? (almost certainly)
 - Can the UKF be made to perform arbitrarily bad? (also yes, but the systems are likely more pathological)

UKF Extensions
 - Increase the number of sigma points
   - more points per shell
   - more shells
 - Hybrid methods
   - something like UKF but we also use 1st order information at each sigma points?
   - EKF but with 2nd order information?

Experiments
 - Direct belief propogration comparisons
 - End-to-end filtering benchmarks (on simulated systems)

## References

 - https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
 - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6057978
 - https://www.cs.unc.edu/~welch/kalman/kalmanPaper.html (Original KF paper)
 - https://www.cs.unc.edu/~welch/kalman/media/pdf/Julier1997_SPIE_KF.pdf (Original UKF paper)

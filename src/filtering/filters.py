import numpy as np
import scipy.linalg

from systems import GaussianSystemModel, SystemModel, LinearSystemModel, DifferentiableSystemModel
from sigma_points import SigmaPointSelector, StandardSigmaPointSelector


class Filter:
    """
    High level representation of the Baye's Filter
    """

    def __init__(self, system: SystemModel):
        self.system = system

    def predict_step(self, u):
        pass

    def update_step(self, z):
        pass


class KalmanFilter(Filter):
    """
    Implementation of the Bayes Filter where state belief is represented by a multivariate gaussian,
    restricted to purely linear system models with additive zero-mean gaussian noise models.
    """
    def __init__(self, system: LinearSystemModel):
        super().__init__(system)
        self.mean = None
        self.covariance = None
    
    def initialize(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
    
    def predict_step(self, u):
        """
        Kalman predict step:
         - µ = Aµ + Bu
         - Σ = AΣA^T + R
        """
        self.mean = self.system.A @ self.mean + self.system.B @ u
        self.covariance = self.system.A @ self.covariance @ self.system.A.T + self.system.R

    def update_step(self, z):
        """
        Kalman update step:
         - µ = µ + K(z - Cµ)
         - Σ = Σ - KCΣ

        where
         - K = ΣC^T(CΣC^T + Q)^-1
        is the so-called "Kalman gain"
        """
        K = self.covariance @ self.system.C.T @ np.linalg.inv(self.system.C @ self.covariance @ self.system.C.T + self.system.Q)

        self.mean += K @ (z - self.system.C @ self.mean)
        self.covariance -= K @ self.system.C @ self.covariance


class ExtendedKalmanFilter(Filter):
    """
    Implementation of the Bayes Filter where state belief is represented by a multivariate gaussian,
    allowing for potentially non-linear systems, with potentially non-additive zero-mean gaussian noise models.

    System model is linearized at the current state estimate, and then linear Kalman estimation is applied.
    """
    def __init__(self, system: DifferentiableSystemModel):  # NOTE: this type hint should be the intersection of DifferentiableSystemModel and GaussianSystemModel, but python doesn't currently support intersection types
        super().__init__(system)
        self.mean = None
        self.covariance = None
    
    def initialize(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
    
    def predict_step(self, u):
        """
        First, linearize the dynamics model at the current state estimate to obtain A, B, and L
        (respectively F_x, F_u, and F_w).
        Then, perform the Kalman predict step:
         - µ = f(µ, u)
         - Σ = AΣA^T + LRL^T
        """
        # linearize dynamics model
        w_mean = np.zeros((self.system.dynamics_noise_dim, 1))
        A, B, L = self.system.query_dynamics_jacobian(self.mean, u, w_mean)

        # Kalman predict step
        self.mean = self.system.query_dynamics_model(self.mean, u)
        self.covariance = A @ self.covariance @ A.T + L @ self.system.dynamics_noise_cov @ L.T

    def update_step(self, z):
        """
        First, linearize the measurement model at the current state estimate to obtain C and M
        (respectively H_x and H_v).
        Then, perform the Kalman update step:
         - µ = µ + K(z - h(µ))
         - Σ = Σ - KCΣ

        where
         - K = ΣC^T(CΣC^T + MQM^T)^-1
        is the so-called "Kalman gain"
        """
        # linearize measurement model
        v_mean = np.zeros((self.system.measurement_noise_dim, 1))
        C, M = self.system.query_measurement_jacobian(self.mean, v_mean)

        # compute Kalman gain
        K = self.covariance @ C.T @ np.linalg.inv(C @ self.covariance @ C.T + M @ self.system.measurement_noise_cov @ M.T)

        # Kalman update state
        self.mean += K @ (z - self.system.query_measurement_model(self.mean))
        self.covariance -= K @ C @ self.covariance

    

class UnscentedKalmanFilter(Filter):
    def __init__(self, system: GaussianSystemModel, sigma_point_selector: SigmaPointSelector = None):
        super().__init__(system)

        self.mean = None
        self.covariance = None

        if sigma_point_selector is None:
            sigma_point_selector = StandardSigmaPointSelector()

        self.sigma_point_selector = sigma_point_selector


    def initialize(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
    
    
    def predict_step(self, u):
        # Note: for non-additive noise models, UKF selects sigma points from an "augmented" state + noise space
        augmented_mean = np.concatenate([self.mean, np.zeros((self.system.dynamics_noise_dim, 1))])
        augmented_cov = scipy.linalg.block_diag(self.covariance, self.system.dynamics_noise_cov)

        # select sigma points in accordance with prior
        sigma_points, weights_mean, weights_cov = self.sigma_point_selector.select_sigma_points(augmented_mean, augmented_cov)

        # pass sigma points through dynamics model
        sigma_points_transformed = []
        for i in range(sigma_points.shape[1]):
            point = sigma_points[:,[i]]
            x, w = np.split(point, [self.system.state_dim])     # un-augment
            x_prime = self.system.query_dynamics_model(x, u, w)
            sigma_points_transformed.append(x_prime)
        
        sigma_points_transformed = np.hstack(sigma_points_transformed)

        # fit a gaussian to weighted transformed sigma points
        mean_hat = np.sum(weights_mean * sigma_points_transformed, axis=1, keepdims=True)
        cov_hat = weights_cov * (sigma_points_transformed - mean_hat) @ (sigma_points_transformed - mean_hat).T

        self.mean = mean_hat
        self.covariance = cov_hat

     
    def update_step(self, z):
        # Note: for non-additive noise models, UKF selects sigma points from an "augmented" state + noise space
        augmented_mean = np.concatenate([self.mean, np.zeros((self.system.measurement_noise_dim, 1))])
        augmented_cov = scipy.linalg.block_diag(self.covariance, self.system.measurement_noise_cov)

        # select sigma points in accordance with prior
        sigma_points, weights_mean, weights_cov = self.sigma_point_selector.select_sigma_points(augmented_mean, augmented_cov)

        # pass sigma points through measurement model
        sigma_points_transformed = []
        for i in range(sigma_points.shape[1]):
            point = sigma_points[:,[i]]
            x, v = np.split(point, [self.system.state_dim])     # un-augment
            x_prime = self.system.query_measurement_model(x, v)
            sigma_points_transformed.append(x_prime)

        sigma_points_transformed = np.hstack(sigma_points_transformed)

        # fit a gaussian to weighted transformed sigma points
        z_hat = np.sum(weights_mean * sigma_points_transformed, axis=1, keepdims=True)

        cov_zz = (weights_cov * (sigma_points_transformed - z_hat)) @ (sigma_points_transformed - z_hat).T
        cov_xz = weights_cov * (sigma_points[:self.system.state_dim, :] - self.mean) @ (sigma_points_transformed - z_hat).T

        # Kalman update step
        K = cov_xz @ np.linalg.inv(cov_zz)
        self.mean += K @ (z - z_hat)
        self.covariance -= K @ cov_zz @ K.T

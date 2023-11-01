import numpy as np
from Systems import SystemModel, LinearSystemModel, DifferentiableSystemModel


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
    def __init__(self, system: DifferentiableSystemModel):
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
        A, B, L = self.system.dynamics_jacobian(self.mean, u, w_mean)

        # Kalman predict step
        self.mean = self.system.dynamics_model(self.mean, u)
        self.covariance = A @ self.covariance @ A.T + L @ self.system.R @ L.T

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
        C, M = self.system.measurement_jacobian(self.mean, v_mean)

        # compute Kalman gain
        K = self.covariance @ C.T @ np.linalg.inv(C @ self.covariance @ C.T + M @ self.system.Q @ M.T)

        # Kalman update state
        self.mean += K @ (z - self.system.measurement_model(self.mean))
        self.covariance -= K @ C @ self.covariance

    

class UnscentedKalmanFilter(Filter):
    pass    # TODO

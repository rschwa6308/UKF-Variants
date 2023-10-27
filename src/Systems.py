import numpy as np


class SystemModel:
    def __init__(self, state_dim, control_dim, measurement_dim):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.measurement_dim = measurement_dim

    def dynamics_model(self, x: np.array, u: np.array) -> np.array:
        """Simulate dynamics at state x with control input u"""
        pass

    def measurement_model(self, x: np.array) -> np.array:
        """Simulate measurement at state x"""
        pass


class LinearSystemModel(SystemModel):
    """
    A time-invariant system model with linear dynamics, linear measurement model, and gaussian noises.

    Dynamics: f(x, u) = Ax + Bu + N(0, R)
    Measurement: h(x) = Cx + N(0, Q)

    Note: different texts use different conventions for these parameters
    """

    def __init__(self, A: np.array, B: np.array, C: np.array, R: np.array, Q: np.array):
        super().__init__(A.shape[0], B.shape[1], C.shape[0])

        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
    
    def dynamics_model(self, x: np.array, u: np.array) -> np.array:
        eps = np.random.multivariate_normal(np.zeros(self.state_dim), self.R).reshape(self.state_dim, 1)
        return self.A @ x + self.B @ u + eps
    
    def measurement_model(self, x: np.array) -> np.array:
        delta = np.random.multivariate_normal(np.zeros(self.measurement_dim), self.Q, check_valid="raise")
        return self.C @ x + delta


class DifferentiableSystemModel(SystemModel):
    def dynamics_jacobian(x):
        pass    # TODO

    def measurement_jacobian(x):
        pass    # TODO

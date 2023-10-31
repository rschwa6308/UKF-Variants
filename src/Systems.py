from typing import Tuple, Callable
import numpy as np

import jax.numpy as jnp
import jax



class SystemModel:
    """
    An abstract class representing a system of interest, including a dynamics model and measurement model.

    Constructor requires the dimensions of the following spaces:
     - x: state space
     - u: control space
     - z: measurement space
     - w: dynamics noise space (optional, defaults to same as state space)
     - v: measurement noise space (optional, defaults to same as measurement space)

    Subclasses include:
     - `LinearSystemModel`
     - `DifferentiableSystemModel`
    """
    def __init__(self, state_dim, control_dim, measurement_dim, dynamics_noise_dim=None, measurement_noise_dim=None):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.measurement_dim = measurement_dim

        if dynamics_noise_dim is None:
            dynamics_noise_dim = self.state_dim
        
        if measurement_noise_dim is None:
            measurement_noise_dim = self.measurement_dim
        
        self.dynamics_noise_dim = dynamics_noise_dim
        self.measurement_noise_dim = measurement_noise_dim

    def dynamics_model(self, x: np.array, u: np.array) -> np.array:
        "Simulate noisy dynamics at state x with control input u"

    def measurement_model(self, x: np.array) -> np.array:
        "Simulate noisy measurement at state x"


class DifferentiableSystemModel(SystemModel):
    """
    An abstract class representing a system model with differentiable dynamics and measurement models.
    Such a system is amenable to the EKF.

    Subclasses include:
     - `LinearSystemModel`: special case where jacobians are determined directly from system parametrization 
     - `AutoDiffSystemModel`: jacobians computed through automatic differentiation
     - `SymbDiffSystemModel`: jacobians provided explicitly be user
    """

    def dynamics_jacobian(self, x: np.array, u: np.array) -> Tuple[np.array, np.array, np.array]:
        "Compute jacobian of dynamics model wrt x, wrt u, and wrt w. Return all three in a tuple."

    def measurement_jacobian(self, x: np.array) -> Tuple[np.array, np.array]:
        "Compute jacobian of measurement model wrt u and wrt v. Return both in a tuple."


class LinearSystemModel(DifferentiableSystemModel):
    """
    A time-invariant system model with linear dynamics, linear measurement model, and additive gaussian noises.

    Dynamics: f(x, u) = Ax + Bu + N(0, R)
    Measurement: h(x) = Cx + N(0, Q)

    Note: different texts use different conventions for these parameters
    """

    def __init__(self, A: np.array, B: np.array, C: np.array, R: np.array, Q: np.array):
        if R.shape[0] != A.shape[0]:
            raise ValueError("Dynamics noise covariance matrix must match state dim (as determined by A.shape)!")

        if Q.shape[0] != C.shape[0]:
            raise ValueError("Measurement noise covariance matrix must match measurement dim (as determined by C.shape)!")

        super().__init__(A.shape[0], B.shape[1], C.shape[0])

        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
    
    def dynamics_model(self, x: np.array, u: np.array) -> np.array:
        w = np.random.multivariate_normal(np.zeros(self.state_dim), self.R).reshape((-1, 1))
        return self.A @ x + self.B @ u + w
    
    def measurement_model(self, x: np.array) -> np.array:
        v = np.random.multivariate_normal(np.zeros(self.measurement_dim), self.Q).reshape((-1, 1))
        return self.C @ x + v
    
    def dynamics_jacobian(self, x: np.array, u: np.array) -> Tuple[np.array, np.array]:
        return (self.A, self.B, self.R)
    
    def measurement_jacobian(self, x: np.array) -> np.array:
        return (self.C, self.Q)


class AutoDiffSystemModel(DifferentiableSystemModel):
    """
    A time-invariant system model with arbitrary dynamics, arbitrary measurement model, and potentially
    non-additive gaussian noises (R, Q). The jacobians are computed at runtime via automatic differentiation (using JAX). 

    The constructor accepts functions for the underlying dynamics and measurement models. Note that (for the sake of consistency) these functions must operate on column vectors:
     - `dynamics_func(x: (1, state_dim), u: (1, control_dim), w: (1, dynamics_noise_dim)) -> (1, state_dim)`
     - `measurement_func(x: (1, state_dim), v: (1, measurement_noise_dim)) -> (1, measurement_dim)`
    
    Moreover, these functions must operate on `jax.numpy` arrays and must use the corresponding methods.
    """

    def __init__(self, state_dim, control_dim, measurement_dim, dynamics_func: Callable, measurement_func: Callable, R: np.array, Q: np.array):
        super().__init__(state_dim, control_dim, measurement_dim, R.shape[0], Q.shape[0])

        self.dynamics_func = dynamics_func
        self.measurement_func = measurement_func

        self.R = R
        self.Q = Q

        self.dynamics_func_dx = jax.jacfwd(self.dynamics_func, argnums=0)       # wrt first arg (x)
        self.dynamics_func_du = jax.jacfwd(self.dynamics_func, argnums=1)       # wrt second arg (u)
        self.dynamics_func_dw = jax.jacfwd(self.dynamics_func, argnums=2)       # wrt second arg (w)

        self.measurement_func_dx = jax.jacfwd(self.measurement_func, argnums=0) # wrt first arg (x)
        self.measurement_func_dv = jax.jacfwd(self.measurement_func, argnums=1) # wrt second arg (v)

    def dynamics_model(self, x: np.array, u: np.array) -> np.array:
        w = np.random.multivariate_normal(np.zeros(self.dynamics_noise_dim), self.R).reshape((-1, 1))
        return self.dynamics_func(x, u, w)
    
    def measurement_model(self, x: np.array) -> np.array:
        v = np.random.multivariate_normal(np.zeros(self.measurement_noise_dim), self.Q).reshape((-1, 1))
        return self.measurement_func(x, v)

    def dynamics_jacobian(self, x: np.array, u: np.array, w: np.array) -> Tuple[np.array, np.array]:
        # evaluate jacobians (JAX requires explicit floats)
        F_x = self.dynamics_func_dx(x.astype(float), u.astype(float), w.astype(float))
        F_u = self.dynamics_func_du(x.astype(float), u.astype(float), w.astype(float))
        F_w = self.dynamics_func_dw(x.astype(float), u.astype(float), w.astype(float))

        # remove extra column-vector dimensions
        F_x = F_x.reshape(self.state_dim, self.state_dim)
        F_u = F_u.reshape(self.state_dim, self.control_dim)
        F_w = F_w.reshape(self.state_dim, self.dynamics_noise_dim)

        return (F_x, F_u, F_w)

    def measurement_jacobian(self, x: np.array, v: np.array) -> np.array:
        # evaluate jacobian (JAX requires explicit floats)
        H_x = self.measurement_func_dx(x.astype(float), v.astype(float))
        H_v = self.measurement_func_dv(x.astype(float), v.astype(float))

        # remove extra column-vector dimensions
        H_x = H_x.reshape((self.measurement_dim, self.state_dim))
        H_v = H_v.reshape((self.measurement_dim, self.measurement_noise_dim))

        return (H_x, H_v)

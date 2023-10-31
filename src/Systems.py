from typing import Tuple, Callable
import numpy as np

import jax.numpy as jnp
import jax



class SystemModel:
    """
    An abstract class representing a system of interest, including a dynamics model and measurement model.

    Subclasses include:
     - `LinearSystemModel`
     - `DifferentiableSystemModel`
    """
    def __init__(self, state_dim, control_dim, measurement_dim):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.measurement_dim = measurement_dim

    def dynamics_model(self, x: np.array, u: np.array) -> np.array:
        "Simulate dynamics at state x with control input u"

    def measurement_model(self, x: np.array) -> np.array:
        "Simulate measurement at state x"


class DifferentiableSystemModel(SystemModel):
    """
    An abstract class representing a system model with differentiable dynamics and measurement models.
    Such a system is amenable to the EKF.

    Subclasses include:
     - `LinearSystemModel`: special case where jacobians are determined directly from system parametrization 
     - `AutoDiffSystemModel`: jacobians computed through automatic differentiation
     - `SymbDiffSystemModel`: jacobians provided explicitly be user
    """

    def dynamics_jacobian(self, x: np.array, u: np.array) -> Tuple[np.array, np.array]:
        "Compute jacobian of dynamics model wrt x, and wrt u. Return both in a tuple."

    def measurement_jacobian(self, x: np.array) -> np.array:
        "Compute jacobian of measurement model wrt u."


class LinearSystemModel(DifferentiableSystemModel):
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
    
    def dynamics_jacobian(self, x: np.array, u: np.array) -> Tuple[np.array, np.array]:
        return (self.A, self.B)
    
    def measurement_jacobian(self, x: np.array) -> np.array:
        return self.C


class AutoDiffSystemModel(DifferentiableSystemModel):
    """
    A system model in which jacobians are computed at runtime via automatic differentiation
    (using JAX).

    The constructor accepts functions for the underlying dynamics and measurement models.
    """

    def __init__(self, state_dim, control_dim, measurement_dim, dynamics_func: Callable, measurement_func: Callable):
        super().__init__(state_dim, control_dim, measurement_dim)

        self.dynamics_func = dynamics_func
        self.measurement_func = measurement_func

        self.dynamics_func_dx = jax.grad()      # TODO
        self.dynamics_func_du = jax.grad()      # TODO

        self.measurement_func_dx = jax.grad()   # TODO

    def dynamics_jacobian(self, x: np.array, u: np.array) -> Tuple[np.array, np.array]:
        F_x = self.dynamics_func_dx(x, u)
        F_u = self.dynamics_func_du(x, u)

        return (F_x, F_u)
    
    def measurement_jacobian(self, x: np.array) -> np.array:
        H_x = self.measurement_func_dx(x)

        return H_x



class DifferentiableSystemModel(SystemModel):
    def __init__(self):
        super().__init__(3, 3, 3) # Assuming placeholder dimension values until reasoned from f and h

    def f(x1,x2):
        return (x2**2,x1**2)
    
    def h(x1,x2):
        return (x2**3,x1**3)
    
    def dynamics_jacobian(self,x: np.array, u:np.array):
        futureTuple = []
        for i in range(0,x.size):
            futureTuple.append(torch.tensor(float(x[i])))

        myNewTuple = tuple(futureTuple)
        #return jacobian(self.f,myNewTuple)
        return self.jacobian_to_np_array(torch.autograd.functional.jacobian(self.f,myNewTuple))    # TODO

    def measurement_jacobian(x):
        pass    # TODO

    def jacobian_to_np_array(self,jacobian):
        array = np.zeros((self.state_dim,self.state_dim))

        for i in range(0,self.state_dim**2):
            local_tuple = jacobian[i%3]
            array[i//3,i%3] = float(local_tuple[i//3])

        return array

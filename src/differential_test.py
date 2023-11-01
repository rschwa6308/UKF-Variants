# defining system functions
def f_new(x1,x2,x3):
    return (x2**2,x3**2,x1**2)

def h_new(x1,x2,x3):
    return(x2**3,x3**3,x1**3)

# my imports
import numpy as np
from matplotlib import pyplot as plt

from Systems import DifferentiableSystemModel

# creating system models
NL_system = DifferentiableSystemModel()

# replacing system functions
NL_system.f = f_new
NL_system.h = h_new

result_f = NL_system.f(1,2,3)
result_h = NL_system.h(1,2,3)

print('g and h Function Values:')
print(result_f)
print(result_h, '\n')

print('My state and input vectors:')
state = np.array([1,2,3])
print('State -> ',state,'\n')
input = np.zeros(state.shape)

jacobian_result_f = NL_system.dynamics_jacobian(state,input)

print('g and h Jacobians:')
print(jacobian_result_f)
print(type(jacobian_result_f))
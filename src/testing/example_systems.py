import numpy as np
import jax.numpy as jnp

import os, sys
from helpers import wrap2pi
sys.path.append(os.path.dirname(__file__) + "/../filtering/")

from systems import LinearSystemModel, AutoDiffSystemModel



###################### Kinematic Car: Linear System #####################
#                                                                       #
#    - state:   [position, velocity]                                    #
#    - control: [acceleration]                                          #
#    - dynamics: f(x, u) = [                                            #
#         position + velocity*dt + (1/2)(acceleration + noise)*dt^2,    #
#         velocity + acceleration*dt                                    #
#     ]                                                                 #
#    - measurement: h(x) = [position + noise]                           #
#                                                                       #
#########################################################################

dt = 0.1
A = np.array([                  # dynamics model
    [1, dt],
    [0, 1]
])
B = np.array([                  # dynamics model
    [0.5 * dt**2],
    [dt]
])
C = np.array([                  # measurement model
    [1, 0]
])

accel_variance = 0.01           # process noise
R = B @ B.T * accel_variance

measurement_variance = 0.3      # measurement noise
Q = np.array([[measurement_variance]])

kinematic_car_linear = LinearSystemModel(A, B, C, R, Q)
kinematic_car_linear.delta_t = dt



################## Kinematic Car: Non-Linear Drag and Turbulence #################
#                                                                                #
#    - state:   [position, velocity]                                             #
#    - control: [acceleration]                                                   #
#    - dynamics: f(x, u) = [                                                     #
#         position + velocity*dt + (1/2)(acceleration + noise_accel)*dt^2,       #
#         velocity + (acceleration - sign(v) * drag)*dt                          #
#     ] where                                                                    #
#         drag = drag_coeff1*velocity + (drag_coeff2 + noise_drag)*velocity^2    #
#    - measurement: h(x) = [position + noise]                                    #
#                                                                                #
##################################################################################

dt = 0.1
drag_coeff1 = 0.0
drag_coeff2 = 0.4

def dynamics_func(x, u, w):
    pos, vel = x.flatten()
    acc = u.flatten()[0]
    noise_acc, noise_drag = w.flatten()
    drag = drag_coeff1*vel + (drag_coeff2 + noise_drag)*vel**2  # <-- non-linear dependence on v
    return jnp.array([
        [pos + vel*dt + 0.5*(acc + noise_acc)*dt**2],
        [vel + (acc - jnp.sign(vel)*drag)*dt]
    ])

def measurement_func(x, v):
    pos, vel = x.flatten()
    noise = v.flatten()[0]
    return jnp.array([
        [pos + noise]
    ])

# dynamics noise covariance
accel_noise_variance = 0.01
drag_noise_variance = 0.001
R = jnp.array([
    [accel_noise_variance, 0],
    [0, drag_noise_variance]
])

# measurement noise covariance
measurement_variance = 0.3
Q = jnp.array([
    [measurement_variance]
])

kinematic_car_nonlinear = AutoDiffSystemModel(2, 1, 1, dynamics_func, measurement_func, R, Q)
kinematic_car_nonlinear.delta_t = dt



###################### 2D SLAM Non-Linear System ###################
#                                                                  #
#    - state:   [pose | landmarks]                                 #
#         - pose = [x, y, theta]                                   #
#         - landmarks = [x1, y1, ..., xn, yn]                      #
#    - control: [rot1, drive, rot2]                                #
#    - dynamics: f(x, u) = [                                       #
#         pose.rotate(rot1+noise)                                  #
#             .translate_forward(drive+noise)                      #
#             .rotate(rot2+noise) |                                #
#         landmarks                                                #
#     ]                                                            #
#    - measurement: h(x) = [                                       #
#         bearing_i+noise, range_i+noise                           #
#         for i in range(len(landmarks))                           #
#     ]                                                            #
#                                                                  #
####################################################################


def generate_SLAM_system(num_landmarks, dt=0.1):
    def dynamics_func(x, u, w):
        px, py, theta = x[0:3,0]
        rot1, drive, rot2 = u[:,0]
        noise_rot1, noise_drive, noise_rot2 = w[:,0]

        # apply first rotation
        theta += rot1 + noise_rot1

        # apply drive
        px += (drive + noise_drive) * jnp.cos(theta)
        py += (drive + noise_drive) * jnp.sin(theta)

        # apply second rotation
        theta += rot2 + noise_rot2

        x_prime = jnp.vstack([px, py, theta, *x[3:]])     # landmarks unchanged
        return x_prime

    def measurement_func(x, v):
        px, py, theta = x[0:3,0]
        landmarks = x[3:,0]

        dx = landmarks[::2] - px
        dy = landmarks[1::2] - py

        # bearings
        bearing_noises = v[::2,0]
        bearings = jnp.arctan2(dy, dx) - theta + bearing_noises
        bearings = wrap2pi(bearings)

        # ranges
        range_noises = v[1::2,0]
        ranges = jnp.sqrt(dx**2 + dy**2) + range_noises

        z = jnp.hstack([*zip(bearings, ranges)]).reshape(-1, 1)
        return z

    # dynamics noise covariance
    rot1_variance  = 0.0001
    drive_variance = 0.001
    rot2_variance  = 0.0001
    R = np.diag([rot1_variance, drive_variance, rot2_variance])

    # measurement noise covariance
    bearing_variance = 0.001
    range_variance = 0.01
    Q = np.diag([bearing_variance, range_variance] * num_landmarks)

    SLAM_nonlinear = AutoDiffSystemModel(
        3 + 2*num_landmarks, 3, 2*num_landmarks,
        dynamics_func, measurement_func, R, Q
    )
    SLAM_nonlinear.delta_t = dt

    return SLAM_nonlinear

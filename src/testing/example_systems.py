import numpy as np
import jax
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
    @jax.jit
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

    @jax.jit
    def measurement_func(x, v):
        px, py, theta = x[0:3,0]
        landmarks = x[3:,0]

        dx = landmarks[::2] - px
        dy = landmarks[1::2] - py

        # bearings
        bearing_noises = v[::2,0]
        bearings = jnp.arctan2(dy, dx) - theta + bearing_noises
        bearings = wrap2pi(bearings)    # wrap to [-pi, pi] so residuals are correct

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




############################### Double Inverted Pendulum on a Cart ###############################
#                                                                                                #
#    - state:   [cart, theta_1, theta_2, cart_vel, theta_1_vel, theta_2_vel]                     #
#    - control: [cart_accel]                                                                     #
#    - dynamics: f(x, u) = lagrangian mechanics (with gravity) + noises                          #
#    - measurement: h(x) = [cart, theta_1] + noises                                              #
#                                                                                                #
#    See: https://digitalrepository.unm.edu/cgi/viewcontent.cgi?article=1131&context=math_etds   #
#                                                                                                #
##################################################################################################

def runge_kutta(f, t0, y0, h):
    k1 = f(t0, y0)
    k2 = f(t0 + h/2, y0 + h/2*k1)
    k3 = f(t0 + h/2, y0 + h/2*k2)
    k4 = f(t0 + h, y0 + h*k3)

    y1 = y0 + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return y1



from jax.numpy import sin, cos

def generate_pendulum_system(
        M=5.0, m1=2.0, m2=3.0,
        L1=2.0, L2=3.0,
        g=9.81, dt=0.001):

    @jax.jit
    def dynamics_func(x, u, w):
        """ We choose:
        l1 = (1/2)*L1
        l1 = (1/2)*L2
        I1 = (1/12)*m1*L1**2
        I2 = (1/12)*m2*L2**2
        """
        def ode(t, x):
            cart, theta_1, theta_2, cart_vel, theta_1_vel, theta_2_vel = x.flatten()

            D = jnp.array([
                [ M+m1+m2,                         ((1/2)*m1+m2)*L1*cos(theta_1),           (1/2)*m2*L2*cos(theta_2)              ],
                [ ((1/2)*m1+m2)*L1*cos(theta_1),   ((1/3)*m1+m2)*L1**2,                     (1/2)*m2*L1*L2*cos(theta_1 - theta_2) ],
                [ (1/2)*m2*L2*cos(theta_2),        (1/2)*m2*L1*L2*cos(theta_1 - theta_2),   (1/3)*m2*L2**2                        ],
            ])

            C = jnp.array([
                [ 0,   -((1/2)*m1+m2)*L1*sin(theta_1)*theta_1_vel,         -(1/2)*m2*L2*sin(theta_2)*theta_2_vel              ],
                [ 0,    0,                                                  (1/2)*m2*L1*L2*sin(theta_1 - theta_2)*theta_2_vel ],
                [ 0,   -(1/2)*m2*L1*L2*sin(theta_1 - theta_2)*theta_1_vel,   0                                                ],
            ])

            g_noisy = g + w[3,0]
            G = jnp.array([
                [0],
                [-(1/2)*(m1+m2)*L1*g_noisy*sin(theta_1)],
                [-(1/2)*m2*g_noisy*L2*sin(theta_2)],
            ])

            H = jnp.array([
                [1],
                [0],
                [0]
            ])

            D_inv = jnp.linalg.inv(D)

            A = jnp.block([
                [ jnp.zeros((3, 3)),   jnp.eye(3) ],
                [ jnp.zeros((3, 3)),   -D_inv @ C ]
            ])

            B = jnp.block([
                [jnp.zeros((3, 1))],
                [D_inv @ H]
            ])

            L = jnp.block([
                [jnp.zeros((3, 1))],
                [-D_inv @ G]
            ])

            # add some friction in the joints as determined by noise vector
            friction_coeffs = w[:3]**2
            joint_1_vel = theta_1_vel
            joint_2_vel = theta_2_vel - theta_1_vel
            drag = jnp.vstack([
                0,
                0,
                0,
                -jnp.sign(cart_vel) * friction_coeffs[0] * cart_vel,
                -jnp.sign(joint_1_vel) * friction_coeffs[1] * joint_1_vel,
                -jnp.sign(joint_2_vel) * friction_coeffs[2] * joint_2_vel,
            ])

            dx_dt = A@x + B@u + L + drag

            return dx_dt

        # # forward euler
        # x_new = x + ode(0, x) * dt

        # runge kutta
        x_new = runge_kutta(ode, 0, x, dt)

        return x_new

    @jax.jit
    def measurement_func(x, v):
        return x[[0, 1],:] + v

    # dynamics noise covariance
    R = jnp.diag(jnp.array([
        1e-12, #0.2,       # cart_vel drag
        1e-12, # 0.0001,       # joint_1_vel drag
        1e-12, # 0.0001,       # joint_2_vel drag
        0.05       # gravity variance
    ]))

    # measurement noise covariance
    Q = jnp.diag(jnp.array([
        0.20,      # cart
        0.10,      # theta_1
    ]))

    double_pendulum = AutoDiffSystemModel(6, 1, 2, dynamics_func, measurement_func, R, Q)
    double_pendulum.delta_t = dt

    return double_pendulum



################## Random Synthetic System #############################
#                                                                      #
#    - state: x in R^n (constrained to a bounded rectangular region)   #
#    - control: R^m                                                    #
#    - dynamics: f(x, u) = a random C^2 smooth function generated      #
#                          via kernel-based interpolation (+ noise)    #
#    - measurement: h(x) = a random C^2 smooth function generated      #
#                          via kernel-based interpolation (+ noise)    #
#                                                                      #
########################################################################

from scipy.interpolate import RegularGridInterpolator

def random_smooth_function(input_space_limits, output_space_limits, axis_steps=5):
    "Generate a random C^2 smooth function by sampling uniformly and then applying cubic interpolation"

    input_dim = len(input_space_limits)
    output_dim = len(output_space_limits)

    input_space_axis_samples = []
    for i in range(input_dim):
        axis_samples = np.linspace(*input_space_limits[i], axis_steps)
        input_space_axis_samples.append(axis_samples)

    domain_shape = (axis_steps,) * input_dim
    
    function_samples = np.zeros(domain_shape + (output_dim,))
    for i in range(output_dim):
        function_samples[..., i] = np.random.uniform(*output_space_limits[i], size=domain_shape)

    interp = RegularGridInterpolator(input_space_axis_samples, function_samples, method="cubic")

    return interp


# def generate_synthetic_system(state_dim, control_dim, measurement_dim, state_space_limits, state_space_steps):
#     state_space_axis = []
#     for i in range(state_dim):
#         state_space_axis.append(jnp.linspace(*state_space_limits[i], state_space_steps[i]))
    
#     control_space_axis = []
#     for i in range(control_dim):
#         control_space_axis.append(jnp.linspace(*control_space_limits[i], control_space_steps[i]))
    

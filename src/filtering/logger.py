import numpy as np
import scipy.linalg

from systems import GaussianSystemModel, SystemModel, LinearSystemModel, DifferentiableSystemModel
from filters import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter

class Logger:

    def __init__(self):
        self.system = system

    def create_logfile(self):
        pass

    def log_state(self, x):
        pass

class KF_Logger(Logger):

    def __init__(self, filter: KalmanFilter):
        super().__init(filter)


class EKF_Logger(Logger):

    def __init__(self, filter: ExtendedKalmanFilter):
        super().__init(filter)

class UKF_Logger(Logger):

    def __init__(self, filter: UnscentedKalmanFilter):
        super().__init(filter)

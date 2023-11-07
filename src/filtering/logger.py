import numpy as np
import scipy.linalg

from systems import GaussianSystemModel, SystemModel, LinearSystemModel, DifferentiableSystemModel
from filters import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter

class Logger:

    def __init__(self, system: SystemModel):
        self.system = system

    def create_logfile(self):
        pass

    def log_state(self, x):
        pass

class KF_Logger(Logger):

    def __init__(self, filter: KalmanFilter, T, logPath):
        system = filter.system

        self.T_max = T
        self.u_dim = system.control_dim
        self.z_dim = system.measurement_dim
        self.x_dim = system.state_dim
        self.cov_shape = filter.covariance.shape
        self.cov_length = self.cov_shape[0]*self.cov_shape[1]
        self.cov_rows = self.cov_shape[0]
        self.cov_cols = self.cov_shape[1]

        self.logFilePath = logPath
        self.file = open(self.logFilePath,'w')

    def initial_write(self,T,ground_truth):
        self.file.write(str(T) + ' ')
        myString = ''
        gt_vec = ground_truth[0,:,0]
        
        for i in range(0,self.x_dim):
            myString += str(float(gt_vec[i]))

            if i is not self.x_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def iteration_write(self,u,z,x,cov,t):
        self.write_u(u,t)
        self.write_z(z,t)
        self.write_x(x,t)
        self.write_cov(cov,t)

    def write_u(self,u,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for i in range(0,self.u_dim):
            myString += str(float(u[i]))

            if i is not self.u_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_z(self,z,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for i in range(0,self.z_dim):
            myString += str(float(z[i]))

            if i is not self.z_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_x(self,x,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for i in range(0,self.x_dim):
            myString += str(float(x[i]))

            if i is not self.x_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_cov(self,cov,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for row in range(0,self.cov_rows):
            for col in range(0,self.cov_cols):
                myString += str(float(cov[row,col]))

                if row is not self.cov_rows-1 and col is not self.cov_cols-1:
                    myString += ' '
                else:
                    #myString += '\n'
                    pass


        self.file.write(myString)


class EKF_Logger(Logger):

    def __init__(self, filter: ExtendedKalmanFilter, T, logPath):
        system = filter.system

        self.T_max = T
        self.u_dim = system.control_dim
        self.z_dim = system.measurement_dim
        self.x_dim = system.state_dim
        self.cov_shape = filter.covariance.shape
        self.cov_length = self.cov_shape[0]*self.cov_shape[1]
        self.cov_rows = self.cov_shape[0]
        self.cov_cols = self.cov_shape[1]

        self.logFilePath = logPath
        self.file = open(self.logFilePath,'w')

    def initial_write(self,T,ground_truth):
        self.file.write(str(T) + ' ' + '\n')
        myString = ''
        gt_vec = ground_truth[0,:,0]
        
        for i in range(0,self.x_dim):
            myString += str(float(gt_vec[i]))

            if i is not self.x_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def iteration_write(self,u,z,x,cov,t):
        self.write_u(u,t)
        self.write_z(z,t)
        self.write_x(x,t)
        self.write_cov(cov,t)

    def write_u(self,u,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for i in range(0,self.u_dim):
            myString += str(float(u[i]))

            if i is not self.u_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_z(self,z,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for i in range(0,self.z_dim):
            myString += str(float(z[i]))

            if i is not self.z_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_x(self,x,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for i in range(0,self.x_dim):
            myString += str(float(x[i]))

            if i is not self.x_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_cov(self,cov,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for row in range(0,self.cov_rows):
            for col in range(0,self.cov_cols):
                myString += str(float(cov[row,col]))

                if row is not self.cov_rows-1 and col is not self.cov_cols-1:
                    myString += ' '
                else:
                    #myString += '\n'
                    pass


        self.file.write(myString)
        

class UKF_Logger(Logger):

    def __init__(self, filter: UnscentedKalmanFilter, T, logPath):
        system = filter.system

        self.T_max = T
        self.u_dim = system.control_dim
        self.z_dim = system.measurement_dim
        self.x_dim = system.state_dim
        self.cov_shape = filter.covariance.shape
        self.cov_length = self.cov_shape[0]*self.cov_shape[1]
        self.cov_rows = self.cov_shape[0]
        self.cov_cols = self.cov_shape[1]

        self.logFilePath = logPath
        self.file = open(self.logFilePath,'w')

    def initial_write(self,T,ground_truth):
        self.file.write(str(T) + ' ' + '\n')
        myString = ''
        gt_vec = ground_truth[0,:,0]
        
        for i in range(0,self.x_dim):
            myString += str(float(gt_vec[i]))

            if i is not self.x_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def iteration_write(self,u,z,x,cov,t):
        self.write_u(u,t)
        self.write_z(z,t)
        self.write_x(x,t)
        self.write_cov(cov,t)

    def write_u(self,u,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for i in range(0,self.u_dim):
            myString += str(float(u[i]))

            if i is not self.u_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_z(self,z,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for i in range(0,self.z_dim):
            myString += str(float(z[i]))

            if i is not self.z_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_x(self,x,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for i in range(0,self.x_dim):
            myString += str(float(x[i]))

            if i is not self.x_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_cov(self,cov,t):
        self.file.write(str(t) + ' ')
        myString = ''

        for row in range(0,self.cov_rows):
            for col in range(0,self.cov_cols):
                myString += str(float(cov[row,col]))

                if row is not self.cov_rows-1 and col is not self.cov_cols-1:
                    myString += ' '
                else:
                    #myString += '\n'
                    pass


        self.file.write(myString)
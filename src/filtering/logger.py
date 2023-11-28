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

class SLAM_Logger(Logger):

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
        self.file.write(str(T) + ' ' + str(int(self.u_dim)) + ' ' + str(int(self.z_dim)) + ' ' + str(int(self.x_dim)) + ' ' + str(int(self.cov_length)) + ' ' + str(int(self.cov_rows)) + ' ' + str(int(self.cov_cols)) + '\n')
        landmarkString = ''
        xString = ''
        yString = ''
        gt_vec = ground_truth[0,:,0]
        
        for i in range(0,self.x_dim):
            landmarkString += str(float(gt_vec[i]))

            if i is not self.x_dim - 1:
                landmarkString += ' '
            else:
                landmarkString += '\n'

        self.file.write(landmarkString)

        gt_vec_x = ground_truth[:,0,0]
        gt_vec_y = ground_truth[:,1,0]

        for i in range(0,int(T)):
            xString += str(float(gt_vec_x[i]))
            yString += str(float(gt_vec_y[i]))

            if i is not gt_vec_x.size - 1:
                xString += ' '
                yString += ' '
            else:
                xString += '\n'
                yString += '\n'

        self.file.write(xString)
        self.file.write(yString)

    def iteration_write(self,u,z,mean,cov,t):
        self.write_u(u,t)
        self.write_z(z,t)
        self.write_mean(mean,t)
        self.write_cov(cov,t)

    def write_u(self,u,t):
        myString = str(int(t)) + ' '

        for i in range(0,self.u_dim):
            myString += str(float(u[i]))

            if i is not self.u_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_z(self,z,t):
        #self.file.write(str(t) + ' ')
        myString = myString = str(t) + ' '

        for i in range(0,self.z_dim):
            myString += str(float(z[i]))

            if i is not self.z_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_mean(self,x,t):
        #self.file.write(str(t) + ' ')
        myString = myString = str(t) + ' '

        for i in range(0,self.x_dim):
            myString += str(float(x[i]))

            if i is not self.x_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_cov(self,cov,t):
        #self.file.write(str(t) + ' ')
        myString = myString = str(t) + ' '

        for row in range(0,self.cov_rows):
            for col in range(0,self.cov_cols):
                myString += str(float(cov[row,col]))

                if row is not self.cov_rows-1 and col is not self.cov_cols-1:
                    myString += ' '
                else:
                    #myString += '\n'
                    pass
                    
        self.file.write(myString + '\n')

class Generic_Logger(Logger):

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
        self.file.write(str(T) + ' ' + str(int(self.u_dim)) + ' ' + str(int(self.z_dim)) + ' ' + str(int(self.x_dim)) + ' ' + str(int(self.cov_length)) + ' ' + str(int(self.cov_rows)) + ' ' + str(int(self.cov_cols)) + '\n')
        landmarkString = ''
        cartString = ''
        theta1String = ''
        theta2String = ''
        gt_vec = ground_truth[0,:,0]
        
        for i in range(0,self.x_dim):
            landmarkString += str(float(gt_vec[i]))

            if i is not self.x_dim - 1:
                landmarkString += ' '
            else:
                landmarkString += '\n'

        self.file.write(landmarkString)

        gt_vec_cart = ground_truth[:,0,0]
        gt_vec_theta1 = ground_truth[:,1,0]
        gt_vec_theta2 = ground_truth[:,2,0]

        for i in range(0,int(T)):
            cartString += str(float(gt_vec_cart[i]))
            theta1String += str(float(gt_vec_theta1[i]))
            theta2String += str(float(gt_vec_theta2[i]))

            if i != (int(T)-1):
                cartString += ' '
                theta1String += ' '
                theta2String += ' '
            else:
                a = 0
        
        cartString += '\n'
        theta1String += '\n'
        theta2String += '\n'

        self.file.write(cartString)
        self.file.write(theta1String)
        self.file.write(theta2String)

    def iteration_write(self,u,z,mean,cov,t):
        self.write_u(u,t)
        self.write_z(z,t)
        self.write_mean(mean,t)
        self.write_cov(cov,t)

    def write_u(self,u,t):
        myString = str(int(t)) + ' '

        for i in range(0,self.u_dim):
            myString += str(float(u[i]))

            if i is not self.u_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_z(self,z,t):
        #self.file.write(str(t) + ' ')
        myString = myString = str(t) + ' '

        for i in range(0,self.z_dim):
            myString += str(float(z[i]))

            if i is not self.z_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_mean(self,x,t):
        #self.file.write(str(t) + ' ')
        myString = myString = str(t) + ' '

        for i in range(0,self.x_dim):
            myString += str(float(x[i]))

            if i is not self.x_dim - 1:
                myString += ' '
            else:
                myString += '\n'

        self.file.write(myString)

    def write_cov(self,cov,t):
        #self.file.write(str(t) + ' ')
        myString = myString = str(t) + ' '

        for row in range(0,self.cov_rows):
            for col in range(0,self.cov_cols):
                myString += str(float(cov[row,col]))

                if row is not self.cov_rows-1 and col is not self.cov_cols-1:
                    myString += ' '
                else:
                    #myString += '\n'
                    pass
                    
        self.file.write(myString + '\n')
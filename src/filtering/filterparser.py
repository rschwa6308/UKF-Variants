import numpy as np
import scipy.linalg

from systems import GaussianSystemModel, SystemModel, LinearSystemModel, DifferentiableSystemModel
from filters import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter

class Parser:

    def __init__(self):
        self.system = system

    def create_logfile(self):
        pass

    def log_state(self, x):
        pass

class KF_Parser(Parser):

    def __init__(self, string):
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


class EKF_Parser(Parser):

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
        self.file.write(str(T) + ' ')
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

            if i is not self.x_dim - 1:
                xString += ' '
                yString += ' '
            else:
                xString += '\n'
                yString += '\n'

        self.file.write(xString)
        self.file.write(yString)



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
        

class UKF_Parser(Parser):

    def __init__(self, filepath):

        self.T_max = 0
        self.u_dim = 0
        self.z_dim = 0
        self.x_dim = 0
        self.cov_shape = 0
        self.cov_length = 0
        self.cov_rows = 0
        self.cov_cols = 0
        
        self.thingsPerTimestep = 4
        self.position_dimension = 2

        self.logFilePath = filepath
        self.file = open(self.logFilePath,'r')

        self.value_grab()

        self.state_ground_truth_0 = np.zeros((1,self.x_dim))
        self.ground_truth_path = np.zeros((self.position_dimension,self.T_max))

        self.u_matrix = np.zeros((self.T_max-1,self.u_dim))
        self.z_matrix = np.zeros((self.T_max-1,self.z_dim))
        self.mean_matrix = np.zeros((self.T_max-1,self.x_dim))

        self.initial_read()

        self.iterative_read()


    def value_grab(self):
        # Reading and storing initial parameters T_max, u_dim, z_dim, x_dim, cov_length, cov_rows, cov_cols
        lineStart = self.file.readline()
        lineList = lineStart.split(' ')
        i = 0

        for x in lineList:
            self.setValue(x,i)
            i = i + 1

    def setValue(self,x,i):
        if i is 0:
            self.T_max = int(x)
        elif i is 1:
            self.u_dim = int(x)
        elif i is 2:
            self.z_dim = int(x)
        elif i is 3:
            self.x_dim = int(x)
        elif i is 4:
            self.cov_length = int(x)
        elif i is 5:
            self.cov_rows = int(x)
        elif i is 6:
            self.cov_cols = int(x)
        else:
            pass

    def initial_read(self):
        # Reading and storing ground truth state at beginning time
        lineStateGroundTruth0 = self.file.readline()
        lineList0 = lineStateGroundTruth0.split(' ')
        i = 0

        for L in lineList0:
            self.state_ground_truth_0[0,i] = float(L)
            i = i + 1

        # Reading and storing ground truth state values of X for T_max timesteps
        lineStateGroundTruthX = self.file.readline()
        lineListX = lineStateGroundTruthX.split(' ')
        i = 0

        thing = np.array(lineListX)
        print(thing.shape)
        for xVal in lineListX:
            self.ground_truth_path[0,i] = float(xVal)
            i = i + 1

        # Reading and storing ground truth state values of Y for T_max timesteps
        lineStateGroundTruthY = self.file.readline()
        lineListY = lineStateGroundTruthY.split(' ')
        i = 0

        for yVal in lineListY:
            self.ground_truth_path[1,i] = float(yVal)
            i = i + 1

    def iterative_read(self):
        for i in range(0,self.T_max-1):
            self.readParameterValues()

    def readParameterValues(self):
        self.read_u()
        self.read_z()
        self.read_mean()
        self.read_cov()

    def read_u(self):
        # Reading and storing input u at a timestep
        lineU = self.file.readline()
        lineListU = lineU.split(' ')
        i = 0
        T = 0

        for U in lineListU:
            if i is 0:
                #print(U)
                T = int(U)
            else:
                self.u_matrix[T-1,i-1] = float(U)
            i = i + 1

    def read_z(self):
        # Reading and storing measurement z at a timestep
        lineZ = self.file.readline()
        lineListZ = lineZ.split(' ')
        i = 0
        T = 0

        for Z in lineListZ:
            if i is 0:
                T = int(Z)
            else:
                self.z_matrix[T-1,i-1] = float(Z)
            i = i + 1

    def read_mean(self):
        # Reading and storing input u at a timestep
        lineMEAN = self.file.readline()
        lineListMEAN = lineMEAN.split(' ')
        i = 0
        T = 0

        for MEAN in lineListMEAN:
            if i is 0:
                T = int(MEAN)
            else:
                self.mean_matrix[T-1,i-1] = float(MEAN)
            i = i + 1
    
    def read_cov(self):
        lineCOV = self.file.readline()

    def return_u(self):
        return self.u_matrix
    
    def return_z(self):
        return self.z_matrix
    
    def return_mean(self):
        return self.mean_matrix
    
    def return_gt0(self):
        return self.state_ground_truth_0
    
    def return_gtp(self):
        return self.ground_truth_path
        

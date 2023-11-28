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

class SLAM_Parser(Parser):

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
        mean_shape = self.mean_matrix.shape
        axis0 = mean_shape[0] + 1
        axis1 = mean_shape[1]
        new_mean_matrix = np.zeros((axis0,axis1))
        new_mean_matrix[0,:] = self.state_ground_truth_0
        new_mean_matrix[1:,:] = self.mean_matrix
        return new_mean_matrix
    
    def return_gt0(self):
        return self.state_ground_truth_0
    
    def return_gtp(self):
        return self.ground_truth_path
    
class Generic_Parser(Parser):

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
        self.position_dimension = 3

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

        # Reading and storing ground truth state values of cartPos for T_max timesteps
        lineStateGroundTruthCartPos = self.file.readline()
        lineListCartPos = lineStateGroundTruthCartPos.split(' ')
        i = 0

        thing = np.array(lineListCartPos)
        print(thing.shape)
        for cartVal in lineListCartPos:
            if cartVal != '':
                self.ground_truth_path[0,i] = float(cartVal)
            i = i + 1

        # Reading and storing ground truth state values of theta1 for T_max timesteps
        lineStateGroundTruthTheta1 = self.file.readline()
        lineListTheta1 = lineStateGroundTruthTheta1.split(' ')
        i = 0

        for theta1 in lineListTheta1:
            if theta1 != '':
                self.ground_truth_path[1,i] = float(theta1)
            i = i + 1

        # Reading and storing ground truth state values of theta2 for T_max timesteps
        lineStateGroundTruthTheta2 = self.file.readline()
        lineListTheta2 = lineStateGroundTruthTheta2.split(' ')
        i = 0

        for theta2 in lineListTheta2:
            if theta2 != '':
                self.ground_truth_path[2,i] = float(theta2)
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
            if U != '':
                if i == 0:
                        T = int(U)
                else:
                    print(T-1)
                    print(i-1)
                    self.u_matrix[T-1,i-1] = float(U)
                i = i + 1

    def read_z(self):
        # Reading and storing measurement z at a timestep
        lineZ = self.file.readline()
        lineListZ = lineZ.split(' ')
        i = 0
        T = 0

        for Z in lineListZ:
            if Z != '':
                if i == 0:
                    if Z != '':
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
            if MEAN != '':
                if i == 0:
                    if MEAN != '':
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
        mean_shape = self.mean_matrix.shape
        axis0 = mean_shape[0] + 1
        axis1 = mean_shape[1]
        new_mean_matrix = np.zeros((axis0,axis1))
        new_mean_matrix[0,:] = self.state_ground_truth_0
        new_mean_matrix[1:,:] = self.mean_matrix
        return new_mean_matrix
    
    def return_gt0(self):
        return self.state_ground_truth_0
    
    def return_gtp(self):
        return self.ground_truth_path
        

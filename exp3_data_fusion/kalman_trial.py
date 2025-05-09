import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, roc_curve, RocCurveDisplay, auc)
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

# const vars
FPS = 10.0 # 10 fps

class kalman_filtering:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=4) # z=measurements=[LKx, LKy, gyroX, gyroY], x=states=[LKx, LKy, gyroX, gyroY,LKx_dot, LKy_dot, gyroX_dot, gyroY_dot]
        self.s = 1 # scaling factor
    def kalman_model(self):
        # Initialize the filter
        # self.kf = KalmanFilter(dim_x=4, dim_z=2, dim_u=2)
        self.kf.F = np.eye(4) # state transition matrix
        self.kf.B = self.s * np.zeros(4,2) # control input matrix
        self.kf.H = np.array([[0.,-1.,0.,0.],[-1.,0.,0.,0.]]) # measurement function
        self.kf.R = np.array([[0.1]]) # measurement noise covariance matrix
        self.kf.Q = Q_discrete_white_noise(dim=2,dt=1/FPS,var=0.1) # process noise covariance matrix

        self.kf.x = np.zeros((4,1)) # initial state estimate
        self.kf.P = np.eye(4) # initial state covariance matrix
        self.kf.z = np.zeros((2,1)) # initial output vector, Z=Hx
        
    # KF - https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html 
    # use this guide - https://medium.com/@satya15july_11937/sensor-fusion-with-kalman-filter-c648d6ec2ec2 
    
    def get_gyro_data(self, i): # 
        pass
    def get_mcp_data(self):
        pass 
    def live_plot(self, prediction, measurement):
        pass
    def main_loop(self, data):
        gyro_measure = data[:,:] # placeholder
        mcp_measure = data[:,0:2] # placeholder
        for i in range(len(data)):
            # u = kalman_filtering.get_mcp_data(i)
            prediction = self.kf.predict(u) # x needs updating too, how does this get updated?? 
            self.kf.z = np.array([kalman_filtering.get_mcp_data(i), kalman_filtering.get_gyro_data(i)])
            new_val = self.kf.update(self.kf.z)
            x = self.kf.x
            kalman_filtering.live_plot(x)

if __name__ == "__main__":
    kf = kalman_filtering()
    try:
        kf.kalman_model()
    except Exception as e:
        # print(f"Error in kalman_model: {e}")
        # raise e("Error in kalman_model: {e}")
        raise SystemError(f"Error in kalman_model: {e}")
    except KeyboardInterrupt:
        raise SystemExit('KeyBoardInterrupt: Exiting the program.')

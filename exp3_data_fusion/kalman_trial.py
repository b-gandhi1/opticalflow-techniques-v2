import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, roc_curve, RocCurveDisplay, auc)
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import time
import glob 
import sys

# const vars
FPS = 10.0 # 10 fps

class kalman_filtering:
    def __init__(self):
        
        self.mode = "none"
        dim_x, dim_z = 8, 4 # state dimension, measurement dimension
        self.dt = 1/FPS # time step
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z) # z=measurements=[LKx, LKy, gyroX, gyroY], x=states=[LKx, LKy, gyroX, gyroY,LKx_dot, LKy_dot, gyroX_dot, gyroY_dot]
        self.s = 1 # scaling factor
        self.kf.F = np.array([[1.,0.,0.,0.,self.dt,0.,0.,0.],
                              [0.,1.,0.,0.,0.,self.dt,0.,0.],
                              [0.,0.,1.,0.,0.,0.,self.dt,0.],
                              [0.,0.,0.,1.,0.,0.,0.,self.dt],
                              [0.,0.,0.,0.,1.,0.,0.,0.],
                              [0.,0.,0.,0.,0.,1.,0.,0.],
                              [0.,0.,0.,0.,0.,0.,1.,0.],
                              [0.,0.,0.,0.,0.,0.,0.,1.],]) # state transition matrix, 8 x 8
        self.kf.H = np.eye(dim_z,dim_x) # measurement function
        self.kf.R = np.eye(dim_z)*0.1 # measurement noise covariance matrix
        self.kf.Q = Q_discrete_white_noise(dim=2,dt=1/FPS,var=0.1) # process noise covariance matrix

        self.kf.x = np.zeros((dim_x,1)) # initial state estimate
        self.kf.P = np.eye(dim_x) # initial state covariance matrix
        self.kf.z = np.zeros((dim_z,1)) # initial output vector, z=measurements
        self.kf._I = np.eye(dim_x)
        
    # KF - https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html 
    # use this guide - https://medium.com/@satya15july_11937/sensor-fusion-with-kalman-filter-c648d6ec2ec2 
    
    def get_data(self, i): # 
        path_pitchroll = "imu-fusion-data/pitchroll_concat2/"
    
        if self.mode == "pitch" or "roll": # or "tz":
            read_mode = self.mode
        else:
            read_mode = "**/"
        csvfiles = glob.glob(path_pitchroll+"/"+read_mode+"*.csv",recursive=True) # reading pitch roll data as specified by mode
        print(csvfiles)
        pitchroll_df = pd.concat((pd.read_csv(f, delimiter=',', usecols={'Franka Rx', 'Franka Ry', 
                                                                    'Pressure (kPa)', 
                                                                    'IMU Rx', 'IMU Ry', 
                                                                    'LKx', 'LKy'}, 
                                                                    dtype={'Franka Rx': float, 
                                                                            'Franka Ry': float, 
                                                                            'Pressure (kPa)': float,
                                                                            'IMY Rx': float, 
                                                                            'IMU Ry': float, 
                                                                            'LKx': float, 
                                                                            'LKy': float}) 
                                                                        for f in csvfiles), axis='index')
        print(pitchroll_df.shape)
        
        # modify below
        if self.mode == "pitch":
            return pitchroll_df.loc(i, ['LKx', 'IMU Rx']) # gnd truth = 'Franka Ry'
        elif self.mode == "roll":
            return pitchroll_df.loc(i, ['LKy', 'IMU Ry']) # gnd truth = 'Franka Rx'
        else:
            raise ValueError("Invalid mode. Choose 'pitch' or 'roll'. Current mode: {}".format(self.mode))
        
    def live_plot(self, prediction, measurement):
        pass
    def main_loop(self, data):
        gyro_measure = data[:,:] # placeholder
        mcp_measure = data[:,0:2] # placeholder
        for i in range(len(data)):
            # u = kalman_filtering.get_mcp_data(i)
            prediction = self.kf.predict() # x needs updating too, how does this get updated?? 
            self.kf.z = np.array([kalman_filtering.get_mcp_data(i), kalman_filtering.get_gyro_data(i)])
            new_val = self.kf.update(self.kf.z)
            x = self.kf.x
            kalman_filtering.live_plot(x)
            time.sleep(1/FPS)

if __name__ == "__main__":
    kf = kalman_filtering()
    kf.mode = sys.argv[1] # input mode: "pitch" | "roll"
    try:
        kf.kalman_model()
    except Exception as e:
        # print(f"Error in kalman_model: {e}")
        # raise e("Error in kalman_model: {e}")
        raise SystemError(f"Error in kalman_model: {e}")
    except KeyboardInterrupt:
        raise SystemExit('KeyBoardInterrupt: Exiting the program.')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
#from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, roc_curve, RocCurveDisplay, auc)
from filterpy.kalman import KalmanFilter#, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import glob 
import sys
import os

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
                              [0.,0.,0.,0.,0.,0.,0.,1.]]) # state transition matrix, 8 x 8. constant velocity model.
        self.kf.H = np.eye(dim_z,dim_x) # measurement function
        self.kf.R = np.eye(dim_z)*0.1 # measurement noise covariance matrix
        self.kf.Q = np.eye(dim_x)*0.1 # process noise init
        #Q_discrete_white_noise(dim=dim_z,dt=1/FPS,var=0.1) # process noise covariance matrix

        self.kf.x = np.zeros((dim_x,1)) # initial state estimate
        self.kf.P = np.eye(dim_x) # initial state covariance matrix
        self.kf.z = np.zeros((dim_z,1)) # initial output vector, z=measurements
        self.kf._I = np.eye(dim_x)
        
    # KF - https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html 
    # use this guide - https://medium.com/@satya15july_11937/sensor-fusion-with-kalman-filter-c648d6ec2ec2 
    
    def get_data(self): # 
        # path_pitchroll = os.path.abspath(os.path.join(os.getcwd(), os.pardir,"imu-fusion-data/pitchroll_concat2"))
        path_pitchroll = os.path.join(os.getcwd(), "imu-fusion-data/pitchroll_concat2")
        if self.mode == "pitch" or "roll": # or "tz":
            read_mode = self.mode
        else:
            read_mode = "**/"
        csvfiles = glob.glob(path_pitchroll+"/"+read_mode+"*.csv",recursive=True) # reading pitch roll data as specified by mode
        print(csvfiles)
        pitchroll_df = pd.read_csv(csvfiles[0], delimiter=',', usecols={'Franka Rx', 'Franka Ry', 
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
        print(pitchroll_df.shape)
        if self.mode == "pitch":
            pitchroll_lk = pitchroll_df.loc[:,'LKx']
            pitchroll_gyro = pitchroll_df.loc[:,'IMU Rx']
            gnd_truth = pitchroll_df.loc[:,'Franka Ry'] 
        elif self.mode == "roll":
            pitchroll_lk = pitchroll_df.loc[:,'LKy']
            pitchroll_gyro = pitchroll_df.loc[:,'IMU Ry']
            gnd_truth = pitchroll_df.loc[:,'Franka Rx']
        else:
            raise ValueError("Invalid mode. Choose 'pitch' or 'roll'. Current mode: {}".format(self.mode))
        
        return pitchroll_lk, pitchroll_gyro, gnd_truth
        
    def live_plot(self, x, gnd_truth, residuals):
        plt.ion()
        # plt.clf()
        plt.subplot(211)
        plt.title("Kalman Filter")
        plt.plot(x[0], x[1], 'ro', label='KF')
        plt.plot(gnd_truth, 'bo', label='Ground Truth')
        plt.xlabel("Index")
        plt.ylabel("Rotation "+self.mode) # pitch or roll
        plt.legend()
        plt.subplot(212)
        plt.title("Residuals")
        plt.plot(residuals[0], residuals[1], 'go')#, label='Residuals')
        plt.xlabel("Index")
        plt.ylabel("Residuals")
        plt.legend()
        plt.pause(1/FPS)
        plt.show()
        
    def main_loop(self):
        lk, gyro, gnd_t = kalman_filtering.get_data(self)
        for i in range(len(lk)):
            # u = kalman_filtering.get_mcp_data(i)
            self.kf.predict() # update x
            # update measurement, z, based on motion type
            if self.mode == "pitch":
                self.kf.z = np.array([[lk[i],0., gyro[i], 0.]]).T
            if self.mode == "roll":
                self.kf.z = np.array([[0., lk[i], gyro[i], 0.]]).T
            gnd_truth = gnd_t[i] # ground truth
            self.kf.update(self.kf.z)
            x = self.kf.x
            residuals = self.kf.y
            kalman_filtering.live_plot(self, x,gnd_truth,residuals)

if __name__ == "__main__":
    kf = kalman_filtering()
    kf.mode = sys.argv[1] # input mode: "pitch" | "roll"
    try:
        kf.main_loop()
    except Exception as e:
        # print(f"Error in kalman_model: {e}")
        # raise e("Error in kalman_model: {e}")
        raise Exception(f"Error in kalman_model: {e}")
    except KeyboardInterrupt:
        raise SystemExit('KeyBoardInterrupt: Exiting the program.')

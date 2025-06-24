import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
#from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, roc_curve, RocCurveDisplay, auc)
from filterpy.kalman import KalmanFilter#, UnscentedKalmanFilter, MerweScaledSigmaPoints
import glob 
import sys
import os
from sklearn.preprocessing import MinMaxScaler

# const vars
FPS = 10.0 # 10 fps

class Kalman_filtering:
    def __init__(self):
        
        self.mode = "none"
        dim_x, dim_z = 4, 4 # state dimension, measurement dimension
        self.dt = 1/FPS # time step
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z) # z=measurements=[LKx, LKy, gyroX, gyroY], x=states=[LKx, LKy, LKx_dot, LKy_dot]

        self.kf.F = np.array([[1.,0.,self.dt,0.],
                              [0.,1.,0.,self.dt],
                              [0.,0.,1.,0.],
                              [0.,0.,0.,1.]]) # state transition matrix, 4 x 4. constant velocity model. constant matrix. 
        self.kf.H = np.eye(dim_z,dim_x) # measurement function
        self.kf.R = np.eye(dim_z)*10 # measurement noise covariance matrix
        self.kf.Q = np.eye(dim_x)*10 # process noise init
        #Q_discrete_white_noise(dim=dim_z,dt=1/FPS,var=0.1) # process noise covariance matrix

        self.kf.x = np.zeros((dim_x,1)) # initial state estimate
        self.kf.P = np.eye(dim_x) # initial state covariance matrix
        self.kf.z = np.zeros((dim_z,1)) # initial output vector, z=measurements
        self.kf._I = np.eye(dim_x)

    # KF - https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html 
    # use this guide - https://medium.com/@satya15july_11937/sensor-fusion-with-kalman-filter-c648d6ec2ec2 
        
    def get_data(self): # 
        # path_pitchroll = os.path.abspath(os.path.join(os.getcwd(), os.pardir,"imu-fusion-data/pitchroll_concat2"))
        path_pitchroll = os.path.join(os.getcwd(), "imu-fusion-data/pitchroll_concat3")
        if self.mode == "pitch" or "roll": # or "tz":
            read_mode = self.mode
        else:
            read_mode = "**/"
        csvfiles = glob.glob(path_pitchroll+"/"+read_mode+"*.csv",recursive=True) # reading pitch roll data as specified by mode
        if not csvfiles: # check non-empty
            raise FileNotFoundError(f"No CSV files found in {path_pitchroll} for mode {self.mode}.")
        # print(csvfiles)
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
        pitchroll_df = pitchroll_df.loc[pitchroll_df.ne(0).all(axis=1)].reset_index(drop=True) # remove rows with zero values
        # breakpoint()
        if self.mode == "pitch":
            trim_start = 600 # trim first 600 rows
            trim_end = 3000
            pitchroll_lk = pitchroll_df.loc[trim_start:trim_end,'LKx'] 
            pitchroll_gyro = pitchroll_df.loc[trim_start:trim_end,'IMU Rx'] * (-1)
            gnd_truth = pitchroll_df.loc[trim_start:trim_end,'Franka Ry'] 
            offset_gnd = -37.9
            offset_gyro = 0.0
        elif self.mode == "roll":
            trim = 600
            pitchroll_lk = pitchroll_df.loc[trim:,'LKy']
            pitchroll_gyro = pitchroll_df.loc[trim:,'IMU Ry']
            gnd_truth = pitchroll_df.loc[trim:,'Franka Rx'] * (-1)
            offset_gnd = 52.0
            offset_gyro = -11.2
            
        else:
            raise ValueError("Invalid mode. Choose 'pitch' or 'roll'. Current mode: {}".format(self.mode))
        
        offset_gnd_dat = gnd_truth.values + offset_gnd
        offset_pitchroll_gyro = pitchroll_gyro.values + offset_gyro
        # minmax scaling MCP data
        min_val = offset_gnd_dat.min() 
        print(f"offset_gnd_dat min val: {min_val}")
        max_val = offset_gnd_dat.max() 
        print(f"offset_gnd_dat max val: {max_val}")
        scaler = MinMaxScaler(feature_range=(min_val,max_val))
        # normalise
        scaled_pitchroll_lk = scaler.fit_transform(pitchroll_lk.values.reshape(-1,1)) # reshape for single feature
        # norm_pitchroll_lk = (pitchroll_lk.values - pitchroll_lk.min())/(pitchroll_lk.max() - pitchroll_lk.min()) * (offset_gnd_dat.max() - offset_gnd_dat.min()) + offset_gnd_dat.min()
        print(f"min max of scaled_pitchroll_lk: {scaled_pitchroll_lk.min()}, {scaled_pitchroll_lk.max()}")
        print(f"sizes: scaled_pitchroll_lk: {scaled_pitchroll_lk.shape}, offset_pitchroll_gyro: {offset_pitchroll_gyro.shape}, offset_gnd_dat: {offset_gnd_dat.shape}")
        # test data, plot
        plt.figure()
        plt.plot(scaled_pitchroll_lk, label='LK')
        plt.plot(offset_pitchroll_gyro, label='Gyro')
        plt.plot(offset_gnd_dat, label='Ground Truth')
        plt.legend()
        plt.title(f"TEST: scaled LK data for {self.mode} mode")
        plt.xlabel("Index")
        plt.ylabel("Degrees")
        plt.show()
        
        return scaled_pitchroll_lk, offset_pitchroll_gyro, offset_gnd_dat
    
    def kf_setup(self, lk, gyro, gnd_t, window): # setup KF, get data    
        # lk, gyro, gnd_t = self.get_data()
        
        # breakpoint()
        if window == "None": 
            start = 0
            end = lk.shape[0] # use all data
        else:
            start = 700 # change this to see different parts of data
            end = start + window
            
        plt.ion()
        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(211)
        line1, = ax1.plot([], [], 'r.', label='KF')
        line2, = ax1.plot([], [], 'b.', label='Ground Truth')
        ax1.set_xlim(start, end)
        ax2 = fig.add_subplot(212)
        line3, = ax2.plot([], [], 'g.', label='Residual pos')
        line4, = ax2.plot([], [], 'm.', label='Residual vel')
        ax2.set_xlim(start, end)
        ax1.set_title("Kalman Filter")
        ax2.set_title("Residuals")
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Rotation "+self.mode) # pitch or roll
        ax1.set_xlim(start, end)
        # ax1.set_ylim(-30, 30) # range for lk and gnd truth
        ax2.set_xlabel("Index")
        ax2.set_ylabel("Residuals")
        ax2.set_xlim(start, end)
        ax2.set_ylim(-10, 10)
        plt.tight_layout()

        x_store, gnd_store, residuals_store_pos, residuals_store_vel, index = [], [], [], [], []
        # index = np.linspace(start, window, num=window-start)
        
        for i in range(start, end):
            # u = kalman_filtering.get_mcp_data(i)
            # self.kf.x, self.kf.P = 
            self.kf.predict(F=self.kf.F,Q=self.kf.Q) # update x
            # update measurement, z, based on motion type
            if self.mode == "pitch":
                self.kf.z = np.array([[float(lk[i]), 0., gyro[i], 0.]]).T
                x_ax, res_ax_pos, res_ax_vel = 2, 0, 2
                ax1.set_ylim(-30, 30)
            elif self.mode == "roll":
                self.kf.z = np.array([[0., float(lk[i]), 0., gyro[i]]]).T
                x_ax, res_ax_pos, res_ax_vel = 1, 1, 3
                ax1.set_ylim(-10, 10)
            else: 
                raise ValueError(f"Invalid mode: {self.mode}. Expected 'pitch' or 'roll'.")
            gnd_truth = gnd_t[i] # ground truth
            # self.kf.x, self.kf.P = 
            self.kf.update(z=self.kf.z, H=self.kf.H, R=self.kf.R) # update state with measurement
            # x = self.kf.x
            residuals = self.kf.y
            # kalman_filtering.live_plot(self, x, gnd_truth, residuals)
            # print('x: ', self.kf.x)
            # print('F: ', self.kf.F)
            # print('z: ', self.kf.z)
            # print('gnd_truth: ', gnd_truth)
            # print('residuals: ', residuals)
            # print('z: ', self.kf.z)
            # print('x: ', self.kf.x)
            x_store = np.append(x_store, self.kf.x[x_ax])
            gnd_store = np.append(gnd_store, gnd_truth)
            residuals_store_pos = np.append(residuals_store_pos, residuals[res_ax_pos])
            residuals_store_vel = np.append(residuals_store_vel, residuals[res_ax_vel])
            index = np.append(index, i)
            line1.set_data(index,x_store)
            line2.set_data(index,gnd_store)
            line3.set_data(index,residuals_store_pos)
            line4.set_data(index,residuals_store_vel)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            # plt.pause(1/FPS)
        ax1.legend()
        ax2.legend()
        plt.ioff() # switch off before show
        plt.show()
        
        return x_store

    def mc_sims_kf_loop(self): # Monte Carlo sim setup: 
        import matplotlib
        matplotlib.use('Agg')
        
        num_sims = 100 # number of simulation runs
        print(f"Running {num_sims} Monte Carlo simulations for {self.mode} mode.")
        lk, gyro, gnd_t = self.get_data()
        diff_abs_sq = np.empty() # store RMSE for each simulation
        
        for i in range(num_sims):
            noise_lk = np.random.normal(0, 0.1, lk.shape) + lk # add noise to LK data
            noise_gyro = np.random.normal(0, 0.1, gyro.shape) + gyro # add noise to gyro data
            x_pred = self.kf_setup(lk=noise_lk, gyro=noise_gyro, gnd_t=gnd_t, window="None") # run KF with noisy data
            
            plt.close('all') # suppress any plt fugures
            
            diff_abs_sq = np.append(diff_abs_sq, np.abs(x_pred - gnd_t)**2) # calculate absolute difference 
            breakpoint()
            print(f"Simulation {i+1}/{num_sims} completed, diff_abs_sq: {diff_abs_sq[-1]}")
        rmse = np.sqrt(1/num_sims * np.sum(diff_abs_sq)) # calculate rmse
        print(f"RMSE for {self.mode} mode: {rmse}")
            
if __name__ == "__main__":
    kf = Kalman_filtering()
    kf.mode = sys.argv[1] # input mode: "pitch" | "roll"
    mc = sys.argv[2] if len(sys.argv) > 2 else "none" # monte carlo simulation mode
    
    try:
        if mc == "mc":
            print(f"Running Monte Carlo simulation for {kf.mode} mode.")
            kf.mc_sims_kf_loop()
        else: 
            print(f"Running Kalman Filter for {kf.mode} mode.")
            lk, gyro, gnd_t = kf.get_data()
            kf.kf_setup(lk=lk, gyro=gyro, gnd_t=gnd_t, window=100)
    except KeyboardInterrupt:
        plt.close('all')
        raise SystemExit('KeyBoardInterrupt: Exiting the program.')

# To run: 
# python kalman_filter_fusion.py pitch|roll mc
# mc - optional for monte carlo simulation. if not provided, single KF model is executed. 
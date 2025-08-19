import matplotlib.pyplot as plt
import numpy as np
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
        self.kf.R = np.diag([1.0,0.3,0.2,1.0])# np.eye(dim_z) # measurement noise covariance matrix
        # q_mcp = Q_discrete_white_noise(dim=2,dt=1/FPS,var=1)
        # q_gyro = Q_discrete_white_noise(dim=2,dt=1/FPS,var=100) # process noise for gyro
        # self.kf.Q = block_diag(q_mcp,q_gyro) #
        self.kf.Q = np.eye(dim_x) # process noise
        print(f"Process noise covariance matrix Q: {self.kf.Q}")
        
        self.kf.x = np.zeros((dim_x,1)) # initial state estimate
        self.kf.P = np.eye(dim_x) # initial state covariance matrix
        self.kf.z = np.zeros((dim_z,1)) # initial output vector, z=measurements
        self.kf._I = np.eye(dim_x)

        self.rmse_len = 0 # init for rmse len
        
    def normalize_vector(self, vector):
        min_value = np.min(vector)
        max_value = np.max(vector)
        
        normalized_vector = (vector - min_value) / (max_value - min_value) * 2 - 1
        
        return normalized_vector
    
    def kf_setup(self, lk, gyro, gnd_t, window): # setup KF, get data    
        start = 0
        end = lk.shape[0] # use all data

        self.rmse_len = end - start # set rmse length
        print(f"RMSE length: {self.rmse_len}")
        # breakpoint()
        plt.ion()
        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(211)
        line1, = ax1.plot([], [], 'r.', label='KF')
        line2, = ax1.plot([], [], 'b.', label='Ground Truth')
        ax1.set_xlim(start/FPS, end/FPS)
        ax2 = fig.add_subplot(212)
        line3, = ax2.plot([], [], 'g.', label='Residual pos')
        line4, = ax2.plot([], [], 'm.', label='Residual vel')
        ax2.set_xlim(start/FPS, end/FPS)
        ax1.set_title("Kalman Filter")
        ax2.set_title("Residuals")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Rotation "+self.mode) # pitch or roll
        ax1.set_xlim(start/FPS, end/FPS)
        # ax1.set_ylim(-30, 30) # range for lk and gnd truth
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Residuals")
        ax2.set_xlim(start/FPS, end/FPS)
        ax2.set_ylim(-10, 10)
        plt.tight_layout()
        
        x_store, gnd_store, residuals_store_pos, residuals_store_vel, ts = [], [], [], [], []
        # index = np.linspace(start, window, num=window-start)
        
        for i in range(start, end):
            self.kf.predict(F=self.kf.F,Q=self.kf.Q) # update x
            # update measurement, z, based on motion type
            if self.mode == "pitch" or self.mode == "none":
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
            self.kf.update(z=self.kf.z, H=self.kf.H, R=self.kf.R) # update state with measurement
            residuals = self.kf.y

            x_store = np.append(x_store, self.kf.x[x_ax])
            gnd_store = np.append(gnd_store, gnd_truth)
            residuals_store_pos = np.append(residuals_store_pos, residuals[res_ax_pos])
            residuals_store_vel = np.append(residuals_store_vel, residuals[res_ax_vel])
            ts = np.append(ts, i/FPS)
            line1.set_data(ts,x_store)
            line2.set_data(ts,gnd_store)
            line3.set_data(ts,residuals_store_pos)
            line4.set_data(ts,residuals_store_vel)

            fig.canvas.draw()
            fig.canvas.flush_events()
            
            print("KF progress: {:.2f}%".format((i-start)/(end-start)*100), end='\r') # progress print
            
        ax1.legend()
        ax2.legend()
        plt.ioff() # switch off before show
        plt.show(block=False) # continue running after plot is shown
        
        return x_store
    
    def performance_metrics(self, kf_preds, lk, gyro, gnd_t, tot): # Monte Carlo sim setup: 
        tot = float(tot[-1])
        
        diff_sq = np.abs(kf_preds - gnd_t)**2
        diff_sq_mcp = np.abs(kf_preds - lk)**2
        diff_sq_gyro = np.abs(kf_preds - gyro)**2
        
        mse = 1/tot * np.sum(diff_sq, axis=1) # mean across all trials
        rmse = np.sqrt(mse) # root mean square error
        mse_mcp = 1/tot * np.sum(diff_sq_mcp, axis=1) # mean square error for mcp data
        rmse_mcp = np.sqrt(mse_mcp) # rmse for mcp data
        mse_gyro = 1/tot * np.sum(diff_sq_gyro, axis=1) # mean square error for gyro data
        rmse_gyro = np.sqrt(mse_gyro) # rmse for gyro data
        
        return mse, rmse, mse_mcp, rmse_mcp, mse_gyro, rmse_gyro
    
    def data_indv_mc_sims(self):
        
        print(f"Running datasets for {self.mode} mode.")
        
        # paths for participant data: 
        polaris_csv_path = ...
        imu_csv_path = ...
        mcp_csv_path = ...
        
        # sort glob in ascending order, then remove the first file from each list due to artefacts in roll1.
        gnd_csv_files = sorted(glob.glob(polaris_csv_path))
        # gnd_csv_files = sorted(glob.glob(polaris_csv_path))
        # gnd_csv_files = gnd_csv_files[1:]
        data_frames_gnd = [pd.read_csv(f, usecols=['roll_x','pitch_y','yaw_z']) for f in gnd_csv_files]
        imu_csv_files = sorted(glob.glob(imu_csv_path))
        # imu_csv_files = imu_csv_files[1:]
        data_frames_imu = [pd.read_csv(f, usecols=['Pressure (kPa)','IMU X','IMU Y','IMU Z']) for f in imu_csv_files]
        mcp_csv_files = sorted(glob.glob(mcp_csv_path))
        # mcp_csv_files = mcp_csv_files[1:]
        data_frames_mcp = [pd.read_csv(f, usecols=['x_vals','y_vals','z_vals']) for f in mcp_csv_files]
        
        for f in data_frames_mcp:
            f['x_vals'] = self.normalize_vector(f['x_vals'])
            f['y_vals'] = self.normalize_vector(f['y_vals'])
            f['z_vals'] = self.normalize_vector(f['z_vals'])
        # init stores for rmse calc after
        lk_store = np.empty([len(data_frames_gnd[0]),len(data_frames_gnd)]) 
        kf_preds_store = lk_store.copy()
        gyro_store = lk_store.copy()
        gnd_store = lk_store.copy() 
        
        ts = np.linspace(0, len(data_frames_gnd[0])/FPS, num=len(data_frames_gnd[0])) # time series for x-axis

        for i in range(len(data_frames_gnd)):
            
            if self.mode == "pitch":
                pitchroll_lk = data_frames_mcp[i].loc[:,'x_vals'] 
                pitchroll_gyro = data_frames_imu[i].loc[:,'IMU X'] 
                gnd_truth = data_frames_gnd[i].loc[:,'roll_x'] # roll_x for LK_*4, pitch_y for LK_*2
                offset_gnd = 37.9
                offset_gyro = 0.0
            elif self.mode == "roll":
                pitchroll_lk = data_frames_mcp[i].loc[:,'y_vals']
                pitchroll_gyro = data_frames_imu[i].loc[:,'IMU Y'] * (-1) 
                gnd_truth = data_frames_gnd[i].loc[:,'pitch_y'] * (-1) # pitch_y for LK_*4, roll_x for LK_*2
                offset_gnd = 46.3
                offset_gyro = 11.75
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
            plt.plot(ts,scaled_pitchroll_lk, label='LK')
            plt.plot(ts,offset_pitchroll_gyro, label='Gyro')
            plt.plot(ts,offset_gnd_dat, label='Ground Truth')
            plt.legend()
            plt.title(f"TEST: scaled LK data for {self.mode} {i+1} mode")
            plt.xlabel("Time (s)")
            plt.ylabel("Degrees")
            plt.show(block=False)
            
            print(f"Running for dataset {i+1}/{len(data_frames_gnd)} for {self.mode} mode.")
            # run KF
            x_pred = self.kf_setup(lk=scaled_pitchroll_lk, gyro=offset_pitchroll_gyro, gnd_t=offset_gnd_dat, window="None") # run KF

            # store vars
            lk_store[:,i] = scaled_pitchroll_lk.flatten() # store lk data
            gyro_store[:,i] = offset_pitchroll_gyro.flatten() # store gyro data
            gnd_store[:,i] = offset_gnd_dat.flatten() # store ground truth data
            kf_preds_store[:,i] = x_pred.flatten() # store KF predictions

        # calculate mean rmse across all datasets
        mse, rmse, mse_mcp, rmse_mcp, mse_gyro, rmse_gyro = self.performance_metrics(kf_preds=kf_preds_store, lk=lk_store, gyro=gyro_store, gnd_t=gnd_store, tot=metrics_range)
        ones = np.ones_like(ts) # for plotting
        plt.figure(figsize=(15, 4))
        plt.plot(ts,rmse,'-b', label='RMSE Gnd')
        plt.plot(ts,ones*np.mean(rmse), '--b')
        # plt.plot(ts,mse, '-c', label='MSE Gnd')
        # plt.plot(ts,ones*np.mean(mse), '--c')
        plt.plot(ts, rmse_mcp, '-g', label='RMSE MCP')
        plt.plot(ts, ones*np.mean(rmse_mcp), '--g')
        # plt.plot(ts, mse_mcp, '-m', label='MSE MCP')
        # plt.plot(ts, ones*np.mean(mse_mcp), '--m')
        plt.plot(ts, rmse_gyro, '-r', label='RMSE Gyro')
        plt.plot(ts, ones*np.mean(rmse_gyro), '--r')
        # plt.plot(ts, mse_gyro, '-y', label='MSE Gyro')
        # plt.plot(ts, ones*np.mean(mse_gyro), '--y')
        
        plt.xlabel("Time (s)")
        plt.ylabel("RMSE (deg)")
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        print(f"Maximum RMSE: {np.max(rmse)}, Minimum RMSE: {np.min(rmse)}")
        print(f"Maximum MSE: {np.max(mse)}, Minimum MSE: {np.min(mse)}")

if __name__ == "__main__":
    kf = Kalman_filtering()
    kf.mode = sys.argv[1] # input mode: "pitch" | "roll"
    ...
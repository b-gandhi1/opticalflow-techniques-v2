import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filterpy.kalman import KalmanFilter
import glob 
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

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
        # self.kf.H = np.eye(dim_z,dim_x) # measurement function
        self.kf.H = np.array([[1.,0.,0.,0.],
                            [0.,1.,0.,0.],
                            [1.,0.,1.,0.],
                            [0.,1.,0.,1.]])
        r1, r2, r3, r4, r5 = np.eye(4), np.zeros((4,4)), np.diag([0.2,0.2,0.6,0.6]), np.diag([0.6,0.6,0.2,0.2]), np.diag([0.2,0.6,0.05,0.2]) # final=r5
        self.kf.R = r5 # measurement noise covariance matrix
        q1, q2, q3, q4, q5 = np.eye(4), np.zeros((4,4)), np.eye(4)*0.1, np.diag([0.1,0.1,0.,0.]), np.diag([0.6, 0.1, 0., 0.]) # final=q5
        self.kf.Q = q5 # process noise covariance matrix
        # self.kf.Q = np.eye(dim_x) # process noise
        print(f"Process noise covariance matrix Q: {self.kf.Q} and observation noise covariance matrix R: {self.kf.R}")
        
        self.kf.x = np.zeros((dim_x,1)) # initial state estimate
        self.kf.P = np.eye(dim_x) # initial state covariance matrix
        self.kf.z = np.zeros((dim_z,1)) # initial output vector, z=measurements
        self.kf._I = np.eye(dim_x)

        self.rmse_len = 0 # init for rmse len

    # KF - https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html 
    # use this guide - https://medium.com/@satya15july_11937/sensor-fusion-with-kalman-filter-c648d6ec2ec2 
        
    def normalize_vector(self, vector):
        min_value = np.min(vector)
        max_value = np.max(vector)
        
        normalized_vector = (vector - min_value) / (max_value - min_value) * 2 - 1
        
        return normalized_vector
    
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
            pressure = pitchroll_df.loc[trim_start:trim_end,'Pressure (kPa)']
        elif self.mode == "roll":
            trim = 600
            pitchroll_lk = pitchroll_df.loc[trim:,'LKy']
            pitchroll_gyro = pitchroll_df.loc[trim:,'IMU Ry'] * (-1) # IMY Ry
            gnd_truth = pitchroll_df.loc[trim:,'Franka Rx'] * (-1)
            offset_gnd = 48.3
            offset_gyro = 2.5
            pressure = pitchroll_df.loc[trim:,'Pressure (kPa)']
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
        
        ts = np.linspace(0, len(scaled_pitchroll_lk)/FPS, num=len(scaled_pitchroll_lk)) # time series for x-axis
        # test data, plot
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(ts,scaled_pitchroll_lk, label='LK')
        plt.plot(ts,offset_pitchroll_gyro, label='Gyro')
        plt.plot(ts,offset_gnd_dat, label='Ground Truth')
        plt.legend()
        plt.title(f"TEST: scaled LK data for {self.mode} mode")
        plt.xlabel("Time (s)")
        plt.ylabel(f"Rotation, {self.mode} (Degrees)")
        plt.subplot(2,1,2)
        plt.plot(ts,pressure)
        plt.xlabel("Time (s)")
        plt.ylabel("Pressure (kPa)")
        plt.tight_layout()
        plt.show(block=False)
        
        print("------------------------------------------------")
        print(f"Pressure and MCP motion correlation: {spearmanr(pressure, scaled_pitchroll_lk, alternative='two-sided', nan_policy='propagate')}")
        print(f"Pressure and gnd truth motion correlation: {spearmanr(pressure, offset_gnd_dat, alternative='two-sided', nan_policy='propagate')}")
        print("------------------------------------------------")
        return scaled_pitchroll_lk, offset_pitchroll_gyro, offset_gnd_dat
    
    def synthetic_data(self):
        # trial for onely 1 run using synthetic data
        print("Generating synthetic data for testing.")
        num_samples = 300
        time = np.linspace(0, num_samples/FPS, num_samples)
        lk = np.sin(2 * np.pi * 0.1 * time) * 20 + np.random.normal(0, 7, num_samples)
        gyro = np.sin(2 * np.pi * 0.1 * time) * 19 + np.random.normal(0, 0.5, num_samples) + 3.0 # 3.0 = drift
        gnd = np.sin(2 * np.pi * 0.1 * time) * 20
        
        plt.figure(figsize=(6, 5))
        plt.plot(time, lk, label='MCP_syn')
        plt.plot(time, gyro, label='Gyro_syn')
        plt.plot(time, gnd, label='Gnd_syn')
        plt.ylim(-40,48)
        plt.xlabel("Time (s)")
        plt.ylabel("Rotation (Degrees)")
        plt.legend(loc='upper right')    
        plt.show(block=False)
        
        return lk, gyro, gnd
    def synthetic_data_mc(self):
        # generate synthetic data for testing kf, 10 monte carlo simulations
        print("Generating synthetic data for testing.")
        num_samples = 300
        time = np.linspace(0, num_samples/FPS, num_samples)
        
        lk_mat = np.empty((num_samples, 10)) # 10 simulations
        gnd_mat = np.empty((num_samples, 10)) 
        gyro_mat = np.empty((num_samples, 10))
        kf_mat = np.empty((num_samples, 10))
        
        for i in range(10):
            lk_mat[:, i] = np.sin(2 * np.pi * 0.1 * time) * 20 + np.random.normal(0, 7, num_samples)
            gyro_mat[:, i] = np.sin(2 * np.pi * 0.1 * time) * 19 + np.random.normal(0, 0.5, num_samples) + 3.0 # 3.0 = drift
            gnd_mat[:, i] = np.sin(2 * np.pi * 0.1 * time) * 20
            # calc kf and store
            kf_mat[:,i] = self.kf_setup(lk=lk_mat[:,i], gyro=gyro_mat[:,i], gnd_t=gnd_mat[:,i], window="None") # run KF
        
        kf.plot_metrics_test(kf_preds_store=kf_mat, lk_store=lk_mat, gyro_store=gyro_mat, gnd_store=gnd_mat)
        
        return lk_mat, gyro_mat, gnd_mat
    
    def kf_setup(self, lk, gyro, gnd_t, window): # setup KF, get data    
        # lk, gyro, gnd_t = self.get_data()
        # breakpoint()
        if window == "None": 
            start = 0
            # end = 50 # investigating periodicity
            end = lk.shape[0] # use all data
        else:
            start = 700 # change this to see different parts of data
            end = start + window
        self.rmse_len = end - start # set rmse length
        print(f"RMSE length: {self.rmse_len}")
        # breakpoint()
        plt.ion()
        fig = plt.figure(figsize=(6, 5))
        ax1 = fig.add_subplot(111) # 111 for single plot, 211 for two plots
        line1, = ax1.plot([], [], 'r.', label='KF pred')
        line2, = ax1.plot([], [], 'b.', label='Ground Truth')
        ax1.set_xlim(start/FPS, end/FPS)
        #----
        # ax2 = fig.add_subplot(212)
        # line3, = ax2.plot([], [], 'g.', label='Residual pos')
        # line4, = ax2.plot([], [], 'm.', label='Residual vel')
        # ax2.set_xlim(start/FPS, end/FPS)
        # ax1.set_title("Kalman Filter")
        # ax2.set_title("Residuals")
        #----
        ax1.set_xlabel("Time (s)")
        if self.mode == "none":
            ax1.set_ylabel("Rotation (Degrees)") # synthetic data
        else:
            ax1.set_ylabel(f"Rotation, {self.mode} (Degrees)") # pitch or roll
        ax1.set_xlim(start/FPS, end/FPS)
        #----
        # ax2.set_xlabel("Time (s)")
        # ax2.set_ylabel("Residuals")
        # ax2.set_xlim(start/FPS, end/FPS)
        # ax2.set_ylim(-10, 10)
        #----
        plt.tight_layout()
        
        x_store, gnd_store, residuals_store_pos, residuals_store_vel, ts = [], [], [], [], []
        # index = np.linspace(start, window, num=window-start)
        
        for i in range(start, end):
            self.kf.predict(F=self.kf.F,Q=self.kf.Q) # update x
            
            # update measurement, z, based on motion type
            # z = [x_mcp, y_mcp x_gyro, y_gyro].T --- format
            if self.mode == "pitch" or self.mode == "none" or self.mode == "test": # y_mcp, y_gyro
                self.kf.z = np.array([[0., float(lk[i]), 0., gyro[i]]]).T
                x_ax, res_ax_pos, res_ax_vel = 1, 1, 3
                ax1.set_ylim(-33, 30)
            elif self.mode == "roll": # x_mcp, x_gyro
                self.kf.z = np.array([[float(lk[i]), 0., gyro[i], 0.]]).T
                x_ax, res_ax_pos, res_ax_vel = 0, 0, 2
                ax1.set_ylim(-8, 1)
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
            # line3.set_data(ts,residuals_store_pos)
            # line4.set_data(ts,residuals_store_vel)

            fig.canvas.draw()
            fig.canvas.flush_events()
            
            print("KF progress: {:.2f}%".format((i-start)/(end-start)*100), end='\r') # progress bar print
            
        ax1.legend(loc='lower left')
        # ax2.legend()
        plt.ioff() # switch off before show
        plt.show(block=False) # continue running after plot is shown
        
        return x_store
    
    def performance_metrics_lk(self, lk, gyro, gnd_t, tot): # calculates rmse relative to mcp values
        tot = float(tot[-1]) # num of samples being used, 3 | 4 | 5, depending on which dataset. 
        
        diff_sq_gyro = np.abs(lk - gnd_t)**2
        diff_sq_gnd = np.abs(lk - gyro)**2
        
        rmse_gyro = np.sqrt(1/tot * np.sum(diff_sq_gyro, axis = 1))
        rmse_gnd = np.sqrt(1/tot * np.sum(diff_sq_gnd, axis = 1))
        
        return rmse_gyro, rmse_gnd

    def performance_metrics(self, kf_preds, lk, gyro, gnd_t, tot): # calculates rmse relative to kf estimates
        tot = float(tot[-1])
        # print(f"Tot var for RMSE calc: {tot}.-------------------------")
        
        diff_sq = np.abs(kf_preds - gnd_t)**2
        diff_sq_mcp = np.abs(kf_preds - lk)**2
        diff_sq_gyro = np.abs(kf_preds - gyro)**2
        
        # Handle both 1D and 2D arrays
        if len(diff_sq.shape) == 1:
            # For 1D arrays (single trial), compute element-wise MSE
            mse = diff_sq  # element-wise squared differences
            mse_mcp = diff_sq_mcp
            mse_gyro = diff_sq_gyro
        else:
            # For 2D arrays (multiple trials)
            mse = 1/tot * np.sum(diff_sq, axis=1) # mean across all trials
            mse_mcp = 1/tot * np.sum(diff_sq_mcp, axis=1) # mean square error for mcp data
            mse_gyro = 1/tot * np.sum(diff_sq_gyro, axis=1) # mean square error for gyro data
            
        rmse = np.sqrt(mse) # root mean square error
        rmse_mcp = np.sqrt(mse_mcp) # rmse for mcp data
        rmse_gyro = np.sqrt(mse_gyro) # rmse for gyro data
        
        return mse, rmse, mse_mcp, rmse_mcp, mse_gyro, rmse_gyro
        
    def data_indv_mc_sims(self):
        
        print(f"Running datasets {self.mode} mode.")
        
        # paths for mannequin datasets: 
        franka_csv_path = 'imu-fusion-data/LK_'+self.mode+'4/*euler_gnd*.csv' # gnd data
        imu_csv_path = 'imu-fusion-data/LK_'+self.mode+'4/fibrescope*.csv' # imu data + pressure sensor data
        mcp_csv_path = 'imu-fusion-data/LK_'+self.mode+'4/imu-fusion-outputs*.csv' # mcp data
        
        skip_rows = 13 # skip first 13 rows from all dataframes
        
        # sort glob in ascending order, then remove the first file from each list due to artefacts in roll1.
        gnd_csv_files = sorted(glob.glob(franka_csv_path))
        # gnd_csv_files = gnd_csv_files[1:]
        data_frames_gnd = [pd.read_csv(f, usecols=['roll_x','pitch_y','yaw_z']) for f in gnd_csv_files]
        imu_csv_files = sorted(glob.glob(imu_csv_path))
        # imu_csv_files = imu_csv_files[1:]
        data_frames_imu = [pd.read_csv(f, usecols=['Pressure (kPa)','IMU X','IMU Y','IMU Z'],skiprows=range(1,skip_rows)) for f in imu_csv_files]
        mcp_csv_files = sorted(glob.glob(mcp_csv_path))
        # mcp_csv_files = mcp_csv_files[1:]
        data_frames_mcp = [pd.read_csv(f, usecols=['x_vals','y_vals','z_vals'],skiprows=range(1,skip_rows)) for f in mcp_csv_files]
        
        for f in data_frames_mcp:
            f['x_vals'] = self.normalize_vector(f['x_vals'])
            f['y_vals'] = self.normalize_vector(f['y_vals'])
            f['z_vals'] = self.normalize_vector(f['z_vals'])
        # breakpoint()
        # init stores for rmse calc after
        lk_store = np.empty([len(data_frames_gnd[0]),len(data_frames_gnd)]) 
        kf_preds_store = lk_store.copy()
        gyro_store = lk_store.copy()
        gnd_store = lk_store.copy() 
        
        ts = np.linspace(0, len(data_frames_gnd[0])/FPS, num=len(data_frames_gnd[0])) # time series for x-axis
        
        metrics_range = range(len(data_frames_gnd))
        for i in metrics_range: # iterate over each dataset
            
            if self.mode == "pitch":
                pitchroll_lk = data_frames_mcp[i].loc[:,'x_vals'] 
                pitchroll_gyro = data_frames_imu[i].loc[:,'IMU X'] 
                gnd_truth = data_frames_gnd[i].loc[:,'roll_x'] # roll_x for LK_*4, pitch_y for LK_*2
                offset_gnd = 37.9
                offset_gyro = 0.0
                ylim = (-40,25)
                pressure = data_frames_imu[i].loc[:,'Pressure (kPa)']
            elif self.mode == "roll":
                pitchroll_lk = data_frames_mcp[i].loc[:,'y_vals']
                pitchroll_gyro = data_frames_imu[i].loc[:,'IMU Y'] * (-1) 
                gnd_truth = data_frames_gnd[i].loc[:,'pitch_y'] * (-1) # pitch_y for LK_*4, roll_x for LK_*2
                offset_gnd = 46.3
                offset_gyro = 11.75
                ylim = (-10,1)
                pressure = data_frames_imu[i].loc[:,'Pressure (kPa)']
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
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, num=f"Dataset {i+1} for {self.mode} mode. Pressure mean: {np.mean(pressure):.3f}, std: {np.std(pressure):.3f}")
            ax1.plot(ts,scaled_pitchroll_lk, label='MCP')
            ax1.plot(ts,offset_pitchroll_gyro, label='Gyro')
            ax1.plot(ts,offset_gnd_dat, label='Ground Truth')
            ax1.set_ylim(ylim)
            ax1.legend()
            # ax1.set_title(f"TEST: scaled LK data for {self.mode} {i+1} mode")
            # ax1.set_xlabel("Time (s)")
            ax1.set_ylabel(f"Rotation, {self.mode} (Degrees)")
            ax2.plot(ts,pressure)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Pressure (kPa)")
            plt.tight_layout()
            plt.show(block=False)
            
            print("------------------------------------------------")
            print(f"Pressure and MCP motion correlation: {spearmanr(pressure, scaled_pitchroll_lk, alternative='two-sided', nan_policy='propagate')}")
            print(f"Pressure and gnd truth motion correlation: {spearmanr(pressure, gnd_truth, alternative='two-sided', nan_policy='propagate')}")
            print("------------------------------------------------")
            
            print(f"Running for dataset {i+1}/{len(data_frames_gnd)} for {self.mode} mode.")
            x_pred = self.kf_setup(lk=scaled_pitchroll_lk, gyro=offset_pitchroll_gyro, gnd_t=offset_gnd_dat, window="None") # run KF

            # store vars
            lk_store[:,i] = scaled_pitchroll_lk.flatten() # store lk data
            gyro_store[:,i] = offset_pitchroll_gyro.flatten() # store gyro data
            gnd_store[:,i] = offset_gnd_dat.flatten() # store ground truth data
            kf_preds_store[:,i] = x_pred.flatten() # store KF predictions

        # rmse relative to kf estimations: 
        mse, rmse, mse_mcp, rmse_mcp, mse_gyro, rmse_gyro = self.performance_metrics(kf_preds=kf_preds_store, lk=lk_store, gyro=gyro_store, gnd_t=gnd_store, tot=metrics_range)
        
        ones = np.ones_like(ts) # for plotting
        plt.figure(f"RMSE relative to KF estimates",figsize=(8, 4))
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
        plt.ylim(-3,15)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        
        # print mean and std dev of rmse values -----------
        print("------------------------------------------------")
        trim=5
        print(f"RMSE values relative to KF estimations from mannequin data, {self.mode} mode:")
        print("         RMSE min | RMSE max | RMSE mean | RMSE std dev")
        print(f"Gnd:     {np.min(rmse[trim:]):.4f}   | {np.max(rmse[trim:]):.4f}  | {np.mean(rmse[trim:]):.4f}   | {np.std(rmse[trim:]):.4f}")
        print(f"MCP:     {np.min(rmse_mcp[trim:]):.4f}   | {np.max(rmse_mcp[trim:]):.4f}  | {np.mean(rmse_mcp[trim:]):.4f}   | {np.std(rmse_mcp[trim:]):.4f}")
        print(f"Gyro:    {np.min(rmse_gyro[trim:]):.4f}   | {np.max(rmse_gyro[trim:]):.4f}  | {np.mean(rmse_gyro[trim:]):.4f}   | {np.std(rmse_gyro[trim:]):.4f}")
        print("------------------------------------------------")
        # rmse relative to mcp measurements: 
        rmse_mcp_gnd, rmse_mcp_gyro = self.performance_metrics_lk(lk=lk_store, gyro=gyro_store, gnd_t=gnd_store, tot=metrics_range)
        
        plt.figure("RMSE relative to MCP",figsize=(8,4))
        plt.plot(ts,rmse_mcp_gnd,'-b', label='RMSE Gnd')
        plt.plot(ts,ones*np.mean(rmse_mcp_gnd), '--b')
        plt.plot(ts,rmse_mcp_gyro,'-r', label='RMSE Gyro')
        plt.plot(ts,ones*np.mean(rmse_mcp_gyro), '--r')
        plt.xlabel("Time (s)")
        plt.ylabel("RMSE (deg)")
        plt.tight_layout()
        plt.legend()
        plt.show(block=False)
        # print(f"Maximum RMSE MCP-Gnd: {np.max(rmse_mcp_gnd)}, Minimum RMSE MCP-Gnd: {np.min(rmse_mcp_gnd)}")
        # print(f"Maximum RMSE MCP-Gyro: {np.max(rmse_mcp_gyro)}, Minimum RMSE MCP-Gyro: {np.min(rmse_mcp_gyro)}")

        print(f"RMSE values relative to MCP measurements from mannequin data, {self.mode} mode:")
        print("         RMSE min | RMSE max | RMSE mean | RMSE std dev")
        print(f"Gnd:     {np.min(rmse_mcp_gnd[trim:]):.4f}   | {np.max(rmse_mcp_gnd[trim:]):.4f}  | {np.mean(rmse_mcp_gnd[trim:]):.4f}   | {np.std(rmse_mcp_gnd[trim:]):.4f}")
        print(f"Gyro:    {np.min(rmse_mcp_gyro[trim:]):.4f}   | {np.max(rmse_mcp_gyro[trim:]):.4f}  | {np.mean(rmse_mcp_gyro[trim:]):.4f}   | {np.std(rmse_mcp_gyro[trim:]):.4f}")
        print("------------------------------------------------")

    def plot_metrics_test(self, kf_preds_store, lk_store, gyro_store, gnd_store):
        metrics_range = range(len(kf_preds_store))
        mse, rmse, mse_mcp, rmse_mcp, mse_gyro, rmse_gyro = self.performance_metrics(kf_preds=kf_preds_store, lk=lk_store, gyro=gyro_store, gnd_t=gnd_store, tot=metrics_range)
        ts = np.linspace(0, kf_preds_store.shape[0]/FPS, num=kf_preds_store.shape[0]) # time series for x-axis
        ones = np.ones_like(ts) # for plotting
        
        plt.figure("RMSE relative to KF estimates",figsize=(8, 4))
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
        # print(f"Maximum RMSE: {np.max(rmse)}, Minimum RMSE: {np.min(rmse)}")
        # print(f"Maximum MSE: {np.max(mse)}, Minimum MSE: {np.min(mse)}")
        
        print("------------------------------------------------")
        trim=5
        print(f"RMSE values relative to KF estimations from mannequin data, {self.mode} mode:")
        print("         RMSE min | RMSE max | RMSE mean | RMSE std dev")
        print(f"Gnd:     {np.min(rmse[trim:]):.4f}   | {np.max(rmse[trim:]):.4f}  | {np.mean(rmse[trim:]):.4f}   | {np.std(rmse[trim:]):.4f}")
        print(f"MCP:     {np.min(rmse_mcp[trim:]):.4f}   | {np.max(rmse_mcp[trim:]):.4f}  | {np.mean(rmse_mcp[trim:]):.4f}   | {np.std(rmse_mcp[trim:]):.4f}")
        print(f"Gyro:    {np.min(rmse_gyro[trim:]):.4f}   | {np.max(rmse_gyro[trim:]):.4f}  | {np.mean(rmse_gyro[trim:]):.4f}   | {np.std(rmse_gyro[trim:]):.4f}")
        print("------------------------------------------------")
        # rmse relative to mcp measurements: 
        rmse_mcp_gnd, rmse_mcp_gyro = self.performance_metrics_lk(lk=lk_store, gyro=gyro_store, gnd_t=gnd_store, tot=metrics_range)        
        print(f"RMSE values relative to MCP measurements from mannequin data, {self.mode} mode:")
        print("         RMSE min | RMSE max | RMSE mean | RMSE std dev")
        print(f"Gnd:     {np.min(rmse_mcp_gnd[trim:]):.4f}   | {np.max(rmse_mcp_gnd[trim:]):.4f}  | {np.mean(rmse_mcp_gnd[trim:]):.4f}   | {np.std(rmse_mcp_gnd[trim:]):.4f}")
        print(f"Gyro:    {np.min(rmse_mcp_gyro[trim:]):.4f}   | {np.max(rmse_mcp_gyro[trim:]):.4f}  | {np.mean(rmse_mcp_gyro[trim:]):.4f}   | {np.std(rmse_mcp_gyro[trim:]):.4f}")
        print("------------------------------------------------")


if __name__ == "__main__":
    kf = Kalman_filtering()
    kf.mode = sys.argv[1] # input mode: "pitch" | "roll"
    mc = sys.argv[2] if len(sys.argv) > 2 else "none" # monte carlo simulation mode
    try:
        if mc == "mc" and kf.mode in ["pitch", "roll"]:
            
            # kf.mc_sims_kf_loop(lk=lk, gyro=gyro, gnd_t=gnd_t)
            kf.data_indv_mc_sims()
        elif kf.mode == 'none' and mc == "test": # test with synthetic data
            lk_syn, gyro_syn, gnd_t_syn = kf.synthetic_data()
            kf_pred = kf.kf_setup(lk=lk_syn, gyro=gyro_syn, gnd_t=gnd_t_syn, window="None")
            kf.plot_metrics_test(kf_preds_store=kf_pred, lk_store=lk_syn, gyro_store=gyro_syn, gnd_store=gnd_t_syn)
        elif kf.mode == "test" and mc == "mc":
            lk_syn, gyro_syn, gnd_t_syn = kf.synthetic_data_mc()
            # kf_pred = kf.kf_setup(lk=lk_syn, gyro=gyro_syn, gnd_t=gnd_t_syn, window="None")
            # kf.plot_metrics_test(kf_preds_store=kf_pred, lk_store=lk_syn, gyro_store=gyro_syn, gnd_store=gnd_t_syn)
            # make a sound when done: 
            print('\a')  # This may not work in some environments
            print("Synthetic data test complete.")
        else:
            lk, gyro, gnd_t = kf.get_data()
            kf.kf_setup(lk=lk, gyro=gyro, gnd_t=gnd_t, window=100)
        print('All plots displayed, press any key to close all windows and exit on a figure window.')
        while True: 
            if plt.waitforbuttonpress():
                plt.close('all')
                break
        raise SystemExit('Button pressed: Closing all windows and terminating the program.')
    except KeyboardInterrupt:
        plt.close('all')
        raise SystemExit('KeyBoardInterrupt: Exiting the program.')

# To run: 
# python kalman_filter_fusion.py pitch|roll mc
# mc - optional for monte carlo simulation. if not provided, single KF model is executed. 
# to run test: 
# python kalman_filter_fusion.py none test
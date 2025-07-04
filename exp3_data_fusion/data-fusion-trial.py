import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import os
import sys
import torch
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score

class DataFusionTrial:
    def __init__(self, mode="none"): # mode for pitch | roll
        self.mode = mode
        self.model = "" # ML model
        self.useimu = "none"
        wandb.init(project="dump", save_code=True) # replace with your wandb entity name
    
    def get_data(self):
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
            pitchroll_lk = pitchroll_df.loc[trim_start:trim_end,['LKx','LKy']] 
            pitchroll_gyro = pitchroll_df.loc[trim_start:trim_end,['IMU Rx','IMU Ry']]
            gnd_truth = pitchroll_df.loc[trim_start:trim_end,['Franka Rx','Franka Ry']] 
            offset_gnd = 37.9
            offset_gyro = 0.0
            lk_ax, gyro_ax, franka_ax = 'LKx', 'IMU Rx', 'Franka Ry'
        elif self.mode == "roll":
            trim = 600
            pitchroll_lk = pitchroll_df.loc[trim:,['LKx','LKy']]
            pitchroll_gyro = pitchroll_df.loc[trim:,['IMU Rx','IMU Ry']]
            gnd_truth = pitchroll_df.loc[trim:,['Franka Rx','Franka Ry']]
            offset_gnd = -48.3
            offset_gyro = -2.5
            lk_ax, gyro_ax, franka_ax = 'LKy', 'IMU Ry', 'Franka Rx'
            
        else:
            raise ValueError("Invalid mode. Choose 'pitch' or 'roll'. Current mode: {}".format(self.mode))
        
        offset_gnd_dat = gnd_truth + offset_gnd
        offset_pitchroll_gyro = pitchroll_gyro + offset_gyro
        # minmax scaling MCP data
        min_val = offset_gnd_dat.min() 
        print(f"offset_gnd_dat min val: {min_val}")
        max_val = offset_gnd_dat.max() 
        print(f"offset_gnd_dat max val: {max_val}")
        scaler = MinMaxScaler(feature_range=(min_val,max_val))
        # normalise
        scaled_pitchroll_lk = scaler.fit_transform(pitchroll_lk.values.reshape(-1,1)) # reshape for single feature
        # convert to dataframe 
        scaled_pitchroll_lk = pd.DataFrame(scaled_pitchroll_lk, columns=['LKx','LKy'])
        breakpoint() # check shape
        # norm_pitchroll_lk = (pitchroll_lk.values - pitchroll_lk.min())/(pitchroll_lk.max() - pitchroll_lk.min()) * (offset_gnd_dat.max() - offset_gnd_dat.min()) + offset_gnd_dat.min()
        print(f"min max of scaled_pitchroll_lk: {scaled_pitchroll_lk.loc[lk_ax].min()}, {scaled_pitchroll_lk.loc[lk_ax].max()}")
        print(f"sizes: scaled_pitchroll_lk: {scaled_pitchroll_lk.shape}, offset_pitchroll_gyro: {offset_pitchroll_gyro.shape}, offset_gnd_dat: {offset_gnd_dat.shape}")
        # test data, plot
        plt.figure()
        plt.plot(scaled_pitchroll_lk.loc[lk_ax], label='LK')
        plt.plot(offset_pitchroll_gyro.loc[gyro_ax], label='Gyro')
        plt.plot(offset_gnd_dat.loc[franka_ax], label='Ground Truth')
        plt.legend()
        plt.title(f"TEST: scaled LK data for {self.mode} mode")
        plt.xlabel("Index")
        plt.ylabel("Degrees")
        plt.show(block=False)
        
        pitchroll_df_new = pd.DataFrame(data={'LKx': scaled_pitchroll_lk.loc['LKx'].values,
                                        'LKy': scaled_pitchroll_lk.loc['LKy'].values,
                                        'IMU Rx': offset_pitchroll_gyro.loc['IMU Rx'].values,
                                        'IMU Ry': offset_pitchroll_gyro.loc['IMU Ry'].values,
                                        'Franka Rx': offset_gnd_dat.loc['Franka Rx'].values,
                                        'Franka Ry': offset_gnd_dat.loc['Franka Ry'].values,
                                        'Pressure (kPa)': pitchroll_df['Pressure (kPa)'].values})
        return pitchroll_df_new # post scaling and offsetting for the selected features, easier to visualise raw data. 
    
    def feature_select(self): 
        pitchroll_df = self.get_data()
        
        if self.useimu == "no-imu" and self.mode == "pitchroll": # without imu, no imu data used
            experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy']]
            ground_truth = pitchroll_df.loc[:,['Franka Rx', 'Franka Ry']]

        elif self.useimu == "no-imu" and self.mode == "pitch":
            experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy']]
            ground_truth = pitchroll_df.loc[:,['Franka Ry']]
        
        elif self.useimu == "no-imu" and self.mode == "roll":
            experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy']]
            ground_truth = pitchroll_df.loc[:,['Franka Rx']]
            
        elif self.useimu == "use-imu" and self.mode == "pitch":
            experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy', 'IMU Rx']]
            ground_truth = pitchroll_df.loc[:,['Franka Ry']]

        elif self.useimu == "use-imu" and self.mode == "roll":
            experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy', 'IMU Ry']]
            ground_truth = pitchroll_df.loc[:,['Franka Rx']]

        elif self.useimu == "use-imu" and self.mode == "pitchroll": # with imu, use imu data
            experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy', 'IMU Rx', 'IMU Ry']]
            ground_truth = pitchroll_df.loc[:,['Franka Rx', 'Franka Ry']]

        else: 
            raise ValueError(f"ERROR: Unrecognised input: {self.useimu}. Expected inputs are imu= no-imu | use-imu ")

        # experimental_data = np.hstack((pitchroll_lk, pitchroll_gyro.values.reshape(-1, 1))) 
        data_trainX, data_testX, data_trainY, data_testY = train_test_split(experimental_data, ground_truth, test_size=0.1, shuffle=True, random_state=42) 

        return data_trainX, data_testX, data_trainY, data_testY
    
    def model1():
        pass
    def model2():
        pass
    def main():
        pass

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "none"
    data_fusion_trial = DataFusionTrial(mode=mode)
    # scaled_pitchroll_lk, offset_pitchroll_gyro, offset_gnd_dat = data_fusion_trial.get_data()

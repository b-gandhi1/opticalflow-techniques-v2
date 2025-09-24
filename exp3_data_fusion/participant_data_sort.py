import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from opticalflow import main as klt_main

class ParticipantDataSort:
    def __init__(self):
        self.pitchroll = "none"
        
    def extract_klt(self,i,j):
        
        vid_paths = glob.glob(f"data_collection_with_franka/B07LabTrials/participant-data/participant{str(i)}/fibrescope-{self.pitchroll}{str(j)}*.mp4")
        
        # if path found with (i,j), good. if not, continue to next iteration
        if vid_paths:
            vid_path = vid_paths[0]
        else: 
            print(f"No video file found for participant {i} with {self.pitchroll} {j}, skipping KLT extraction.")
            return 
        
        if not vid_path:
            raise FileNotFoundError(f"No video file found for participant {i} with {self.pitchroll} {j}.")
                
        pitchrollN = self.pitchroll+str(j)
        savefilename = f"participant_data/part{str(i)}"
        klt_main('f',vid_path,pitchrollN,self.pitchroll,j,savefilename)
        
    def plot_polaris_imu(self,i,j):
        file_paths = glob.glob(f"data_collection_with_franka/B07LabTrials/participant-data/participant{str(i)}/fibrescope-{self.pitchroll}{str(j)}*.csv")
        if file_paths:
            file_path = file_paths[0]
        else:
            print(f"No CSV file found for participant {i} with {self.pitchroll} {j}, skipping plotting.")
            return None, None, None
        if not file_path:
            raise FileNotFoundError(f"No CSV file found for participant {i} with {self.pitchroll} {j}.")
        imu_df = pd.read_csv(file_path,usecols=['IMU X', 'IMU Y', 'IMU Z'])
        polaris_df = pd.read_csv(file_path,usecols=['Polaris Rx', 'Polaris Ry', 'Polaris Rz'])
        pressures_df = pd.read_csv(file_path,usecols=['Pressure (kPa)'])
        # plot figures to check data
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(imu_df['IMU X'], label='IMU X')
        # plt.plot(imu_df['IMU Y'], label='IMU Y')
        # plt.plot(imu_df['IMU Z'], label='IMU Z')
        # plt.title(f'IMU Data - Participant {i} {self.pitchroll} {j}')
        # plt.xlabel('Time')
        # plt.ylabel('IMU Values')
        # plt.legend()

        # plt.subplot(1, 2, 2)
        # plt.plot(polaris_df['Polaris Rx'], label='Polaris Rx')
        # plt.plot(polaris_df['Polaris Ry'], label='Polaris Ry')
        # plt.plot(polaris_df['Polaris Rz'], label='Polaris Rz')
        # plt.title(f'Polaris Data - Participant {i} {self.pitchroll} {j}')
        # plt.xlabel('Time')
        # plt.ylabel('Polaris Values')
        # plt.legend()

        # plt.tight_layout()
        
        return imu_df, polaris_df, pressures_df
    
    def transform_polaris_imu(self, imu_df, polaris_df, pressures_df, i, j):
        # Transform polaris and imu data using offset and any necessary rotations, and select apt axes
        if imu_df is None or polaris_df is None:
            print(f"Skipping transformation for participant {i} {self.pitchroll} {j} due to missing data.")
            return

        if self.pitchroll == "pitch":
            imu_df_fin = imu_df['IMU X'] - imu_df['IMU X'].iloc[:5].mean()
            polaris_df_fin = polaris_df['Polaris Rz'] * (-1) - (polaris_df['Polaris Rz'].iloc[:5] * (-1)).mean()
            pressures_motion = pressures_df['Pressure (kPa)']
        elif self.pitchroll == "roll":
            imu_df_fin = imu_df['IMU Y'] - imu_df['IMU Y'].iloc[:5].mean()
            polaris_df_fin = polaris_df['Polaris Ry'] - polaris_df['Polaris Ry'].iloc[:5].mean()
            pressures_motion = pressures_df['Pressure (kPa)']
        else:
            raise ValueError("Invalid pitchroll value. Use 'pitch' or 'roll'.")
        
        # plot, check data
        plt.figure(figsize=(10, 5))
        plt.plot(imu_df_fin, label=imu_df.columns[0])
        plt.plot(polaris_df_fin, label=polaris_df.columns[0])
        plt.title(f'Polaris and IMU Data - Participant {i} {self.pitchroll} {j}')
        plt.xlabel('Time')
        plt.ylabel('Degrees')
        plt.legend()
        
        # save the dataframes
        # imu_pol_df = pd.concat([imu_df_fin, polaris_df_fin], axis=1)
        # imu_pol_df.to_csv(f"participant_data/part{str(i)}/imu_pol_{self.pitchroll}_{j}.csv", index=False)

        return pressures_motion
        
    def main(self):
        pressures_pitch, pressures_roll = [], []
        # self.pitchroll = "pitch"
        # self.transform_polaris_imu(*self.plot_polaris_imu(1,2),1,2)
        # self.pitchroll = "roll"
        # self.transform_polaris_imu(*self.plot_polaris_imu(1,3),1,3)
        
        for i in range(1,10+1):
            for j in range(1,3+1):
                # obtain KLT files and save them: 
                self.pitchroll = "pitch"
                # self.extract_klt(i,j)
                imu_df_pitch, polaris_df_pitch, pressures_df_pitch = self.plot_polaris_imu(i,j)
                if pressures_df_pitch is not None:
                    pressure_dat_pit = self.transform_polaris_imu(imu_df_pitch, polaris_df_pitch, pressures_df_pitch, i, j)
                    pressures_pitch.append(pressure_dat_pit)

                self.pitchroll = "roll"
                # self.extract_klt(i,j)
                imu_df_roll, polaris_df_roll, pressures_df_roll = self.plot_polaris_imu(i,j)
                if pressures_df_roll is not None:
                    pressure_dat_rol = self.transform_polaris_imu(imu_df_roll, polaris_df_roll, pressures_df_roll, i, j)
                    pressures_roll.append(pressure_dat_rol)

        # plot all pressure data for pitch and roll
        plt.figure(figsize=(10, 4))
        trim = 2
        ts = np.linspace(0,60,len(pressures_pitch[0])-trim)
        plt.subplot(2, 1, 1)
        plt.plot(ts, pressures_pitch[0][trim:], ts, pressures_pitch[1][trim:], ts, pressures_pitch[2][trim:])
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (kPa), pitch')
        
        plt.subplot(2, 1, 2)
        plt.plot(ts, pressures_roll[0][trim:], ts, pressures_roll[1][trim:], ts, pressures_roll[2][trim:])
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (kPa), roll')
        
        plt.tight_layout()
                
        setpoint = 1.7
        print(f"control: overall mean pressure: pitch - {np.mean(np.mean(pressures_pitch, axis=0))}, roll - {np.mean(np.mean(pressures_roll, axis=0))}")
        print(f"Max deviation pitch: {np.abs(np.max(np.mean(pressures_pitch, axis=0)[trim:]) - setpoint)}")
        print(f"Min deviation pitch: {np.abs(np.min(np.mean(pressures_pitch, axis=0)[trim:]) - setpoint)}")
        print(f"Max deviation roll: {np.abs(np.max(np.mean(pressures_roll, axis=0)[trim:]) - setpoint)}")
        print(f"Min deviation roll: {np.abs(np.min(np.mean(pressures_roll, axis=0)[trim:]) - setpoint)}")
        # plt.show() # show all figs
        
        # save pressure_pitch and pressure_roll dfs to csv
        pressure_pitch_df = pd.concat([pressures_pitch[0],pressures_pitch[1],pressures_pitch[2]], axis=1)
        pressure_roll_df = pd.concat([pressures_roll[0],pressures_roll[1],pressures_roll[2]], axis=1)
        # print(np.shape(pressures_pitch), np.shape(pressures_roll))
        print(pressure_pitch_df.shape, pressure_roll_df.shape)
        # pressure_pitch_df.to_csv(f"participant_data/part_pressure_pitch.csv", index=False)
        # pressure_roll_df.to_csv(f"participant_data/part_pressure_roll.csv", index=False)


if __name__ == "__main__":
    participant_data_sort = ParticipantDataSort()
    try: 
        participant_data_sort.main()
        while True: 
            if plt.waitforbuttonpress():
                plt.close('all')
                break
        raise SystemExit('Exit Button Pressed: Closing all windows and terminating the program.')
    except KeyboardInterrupt:
        SystemExit("Exiting participant data sort script due to keyboard interrupt.")


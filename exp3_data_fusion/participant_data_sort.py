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
            return None, None
        if not file_path:
            raise FileNotFoundError(f"No CSV file found for participant {i} with {self.pitchroll} {j}.")
        imu_df = pd.read_csv(file_path,usecols=['IMU X', 'IMU Y', 'IMU Z'])
        polaris_df = pd.read_csv(file_path,usecols=['Polaris Rx', 'Polaris Ry', 'Polaris Rz'])
        
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
        
        return imu_df, polaris_df
    
    def transform_polaris_imu(self, imu_df, polaris_df,i,j):
        # Transform polaris and imu data using offset and any necessary rotations, and select apt axes

        if imu_df is None or polaris_df is None:
            print(f"Skipping transformation for participant {i} {self.pitchroll} {j} due to missing data.")
            return

        if self.pitchroll == "pitch":
            imu_df_fin = imu_df['IMU X'] - imu_df['IMU X'].iloc[:5].mean()
            polaris_df_fin = polaris_df['Polaris Rz'] * (-1) - (polaris_df['Polaris Rz'].iloc[:5] * (-1)).mean()
        elif self.pitchroll == "roll":
            imu_df_fin = imu_df['IMU Y'] - imu_df['IMU Y'].iloc[:5].mean()
            polaris_df_fin = polaris_df['Polaris Ry'] - polaris_df['Polaris Ry'].iloc[:5].mean()
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
        imu_pol_df = pd.concat([imu_df_fin, polaris_df_fin], axis=1)
        imu_pol_df.to_csv(f"participant_data/part{str(i)}/imu_pol_{self.pitchroll}_{j}.csv", index=False)

    def main(self):
        
        # self.pitchroll = "pitch"
        # self.transform_polaris_imu(*self.plot_polaris_imu(1,2),1,2)
        # self.pitchroll = "roll"
        # self.transform_polaris_imu(*self.plot_polaris_imu(1,3),1,3)
        
        for i in range(1,10+1):
            for j in range(1,3+1):
                # obtain KLT files and save them: 
                self.pitchroll = "pitch"
                self.extract_klt(i,j)
                self.transform_polaris_imu(*self.plot_polaris_imu(i,j),i,j)
        
                self.pitchroll = "roll"
                self.extract_klt(i,j)
                self.transform_polaris_imu(*self.plot_polaris_imu(i,j),i,j)
        plt.show()


if __name__ == "__main__":
    participant_data_sort = ParticipantDataSort()
    try: 
        participant_data_sort.main()
        while True: 
            if plt.waitforbuttonpress():
                plt.close('all')
                break
        raise SystemExit('Button pressed: Closing all windows and terminating the program.')
    except KeyboardInterrupt:
        SystemExit("Exiting participant data sort script due to keyboard interrupt.")


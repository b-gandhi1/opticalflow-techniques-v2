import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from opticalflow import main as klt_main
from opticalflow import inp_space

class ParticipantDataSort:
    def __init__(self):
        self.pitchroll = "none"
        
    def extract_klt(self,i,j):
        vid_path = glob.glob(f"data_collection_with_franka/B07LabTrials/participant-data/participant{str(i)}/fibrescope-{self.pitchroll}{str(j)}*.mp4")[0]
        if not vid_path:
            raise FileNotFoundError(f"No video file found for participant {i} with {self.pitchroll} {j}.")
        pitchrollN = self.pitchroll+str(j)
        savefilename = f"participant_data/part{str(i)}"
        klt_main('f',vid_path,pitchrollN,self.pitchroll,j,savefilename)
        
    def plot_polaris_imu(self,i,j):
        file_path = glob.glob(f"data_collection_with_franka/B07LabTrials/participant-data/participant{str(i)}/fibrescope-{self.pitchroll}{str(j)}*.csv")[0]
        if not file_path:
            raise FileNotFoundError(f"No CSV file found for participant {i} with {self.pitchroll} {j}.")
        imu_df = pd.read_csv(file_path,columns=['IMU X', 'IMU Y', 'IMU Z'])
        polaris_df = pd.read_csv(file_path,columns=['Polaris Rx', 'Polaris Ry', 'Polaris Rz'])
        
        # plot figures to check data
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(imu_df['IMU X'], label='IMU X')
        plt.plot(imu_df['IMU Y'], label='IMU Y')
        plt.plot(imu_df['IMU Z'], label='IMU Z')
        plt.title(f'IMU Data - Participant {i} {self.pitchroll} {j}')
        plt.xlabel('Time')
        plt.ylabel('IMU Values')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(polaris_df['Polaris Rx'], label='Polaris Rx')
        plt.plot(polaris_df['Polaris Ry'], label='Polaris Ry')
        plt.plot(polaris_df['Polaris Rz'], label='Polaris Rz')
        plt.title(f'Polaris Data - Participant {i} {self.pitchroll} {j}')
        plt.xlabel('Time')
        plt.ylabel('Polaris Values')
        plt.legend()

        plt.tight_layout()

    def main(self):
        
        # inp_space = "1-0" # set spacing for participant study, KLT extraction.
        
        for i in range(1,10+1):
            for j in range(1,2+1):
            
                # # obtain KLT files and save them: 
                self.pitchroll = "pitch"
                # self.extract_klt(i,j)
                self.plot_polaris_imu(i,j)
                
                self.pitchroll = "roll"
                # self.extract_klt(i,j)
                self.plot_polaris_imu(i,j)
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


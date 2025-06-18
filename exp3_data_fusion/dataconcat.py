import pandas as pd
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

class Dataconcat():
    def __init__(self, save, pitchroll):
        self.save = save
        self.pitchroll = pitchroll
        
    def normalize_vector(self, vector):
        min_value = np.min(vector)
        max_value = np.max(vector)
        
        normalized_vector = (vector - min_value) / (max_value - min_value) * 2 - 1
        
        return normalized_vector

    def concatenate_csv_files(self, pitchroll, gnd_path, imu_path, mcp_path):
        gnd_csv_files = glob.glob(gnd_path)
        data_frames_gnd = [pd.read_csv(f, usecols=['roll_x','pitch_y','yaw_z']) for f in gnd_csv_files]
        imu_csv_files = glob.glob(imu_path)
        data_frames_imu = [pd.read_csv(f, usecols=['Pressure (kPa)','IMU X','IMU Y','IMU Z']) for f in imu_csv_files]
        mcp_csv_files = glob.glob(mcp_path)
        data_frames_mcp = [pd.read_csv(f, usecols=['x_vals','y_vals','z_vals']) for f in mcp_csv_files]
        # tz_files = glob.glob(Tz_path)
        # data_frames_tz = [pd.read_csv(f, usecols=['Franka Tz']) for f in tz_files]
        
        # normalise columns in data_frames_mcp
        for f in data_frames_mcp:
            f['x_vals'] = self.normalize_vector(f['x_vals'])
            f['y_vals'] = self.normalize_vector(f['y_vals'])
            f['z_vals'] = self.normalize_vector(f['z_vals'])
        
        concatenated_mcp = pd.concat(data_frames_mcp, ignore_index=True, axis='index')
        # print(concatenated_mcp.shape)
        
        concatenated_gnd = pd.concat(data_frames_gnd, ignore_index=True, axis='index')
        # print(concatenated_imu.shape)
        
        # concatenated_tz = pd.concat(data_frames_tz, ignore_index=True, axis='index')
        # print(concatenated_tz.shape)
        
        concatenated_imu = pd.concat(data_frames_imu, ignore_index=True, axis='index')

        # test plots
        xs = concatenated_gnd.loc[:,['roll_x']]
        ys = concatenated_gnd.loc[:,['pitch_y']]
        ts = np.linspace(0,300,len(xs))
        
        plt.figure()
        
        plt.subplot(3,1,1)
        plt.plot(ts, xs, label='franka_Rx')
        plt.plot(ts, ys, label='franka_Ry')
        plt.legend()
        plt.title(pitchroll)

        plt.subplot(3,1,2)
        ax1 = plt.gca()   
        LKx = concatenated_mcp.loc[:,['x_vals']]
        LKy = concatenated_mcp.loc[:,['y_vals']]
        ax1.plot(ts, LKx, 'b-', label='LKx')
        ax1.set_ylabel('LKx', color='b')
        ax1.tick_params('y', colors='b')
        # ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.plot(ts, LKy, 'g-', label='LKy')
        ax2.set_ylabel('LKy', color='g')
        ax2.tick_params('y', colors='g')
        # ax2.legend(loc='upper right') 
        
        plt.subplot(3,1,3)
        imuX = concatenated_imu.loc[:, ['IMU X']]
        imuY = concatenated_imu.loc[:, ['IMU Y']]
        plt.plot(ts, imuX, label='IMU X')
        plt.plot(ts, imuY, label='IMU Y')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        
        # concatenated_df = pd.concat([concatenated_tz, concatenated_gnd, concatenated_imu, concatenated_mcp], ignore_index=False, axis='columns')
        concatenated_df = pd.concat([concatenated_gnd, concatenated_imu, concatenated_mcp], ignore_index=False, axis='columns')
        # print(concatenated_df.shape)
        
        # remove zeros
        df_nozeros = concatenated_df.loc[concatenated_gnd.ne(0).all(axis=1)].reset_index(drop=True) # remove rows with zero values
        
        # concatenated_df.columns = ['Franka Tz','Franka Rx','Franka Ry','Franka Rz','Pressure (kPa)','IMU Rx','IMU Ry','IMU Rz','LKx','LKy','BrtZ']
        df_nozeros.columns = ['Franka Rx','Franka Ry','Franka Rz','Pressure (kPa)','IMU Rx','IMU Ry','IMU Rz','LKx','LKy','BrtZ']
        
        # # Pair and concatenate each pair of data frames
        # concatenated_pairs = [pd.concat([imu_df, mcp_df], ignore_index=True) for imu_df, mcp_df in zip(data_frames_imu, data_frames_mcp)]
        
        # # Concatenate all pairs into a single data frame
        # concatenated_df = pd.concat(concatenated_pairs, ignore_index=True)
        
        # print sprearman's correlations: 
        
        
        return df_nozeros

    def concatenate_tensor_files(self, path_pattern, skip_entries=15):
        tensor_files = glob.glob(path_pattern)
        tensors = [torch.load(f)[skip_entries:] for f in tensor_files]
        
        # print a sample
        # for tensor in tensors:
        #     print('size: ', tensor.size())
        #     print(tensor[0])
            
        concatenated_tensor = torch.cat(tensors, dim=0)
        return concatenated_tensor

    def main(self, pitchroll):
        franka_csv_path = 'imu-fusion-data/LK_'+pitchroll+'2/*euler_gnd*.csv' # gnd data
        # franka_tz_path = 'imu-fusion-data/LK_'+pitchroll+'2/fibrescope*.csv' # Tz gnd data
        imu_csv_path = 'imu-fusion-data/LK_'+pitchroll+'2/fibrescope*.csv' # imu data + pressure sensor data
        mcp_csv_path = 'imu-fusion-data/LK_'+pitchroll+'2/imu-fusion-outputs*.csv' # mcp data
        tensorX_path_pattern = 'imu-fusion-data/LK_'+pitchroll+'2/tensor_x*.pt' # tensor_x data
        tensorY_path_pattern = 'imu-fusion-data/LK_'+pitchroll+'2/tensor_y*.pt' # tensor_y data
        
        # concatenated_csv = concatenate_csv_files(pitchroll=pitchroll, gnd_path = franka_csv_path, Tz_path = franka_tz_path, imu_path=imu_csv_path, mcp_path=mcp_csv_path)
        concatenated_csv = self.concatenate_csv_files(pitchroll=pitchroll, gnd_path = franka_csv_path, imu_path=imu_csv_path, mcp_path=mcp_csv_path)
        concatenated_tensor_x = self.concatenate_tensor_files(tensorX_path_pattern)
        concatenated_tensor_y = self.concatenate_tensor_files(tensorY_path_pattern)
        
        if self.save == "save": 
            print("Saving concatenated data...")
            concatenated_csv.to_csv('imu-fusion-data/pitchroll_concat3/'+pitchroll+'_concatenated_data.csv', index=False)
            torch.save(concatenated_tensor_x, 'imu-fusion-data/pitchroll_concat3/'+pitchroll+'_concatenated_tensor_x.pt')
            torch.save(concatenated_tensor_y, 'imu-fusion-data/pitchroll_concat3/'+pitchroll+'_concatenated_tensor_y.pt')
        else:
            print("Not saving concatenated data.") 
if __name__ == "__main__":
    try: 
        save = sys.argv[1] if len(sys.argv) > 1 else "no_save"
        concat = Dataconcat(save=save, pitchroll=None)
        concat.main(pitchroll="pitch")
        concat.main(pitchroll="roll")

        # main(pitchroll="tz") # this was not performed, due to mannequin head being too heavy for robot arm! 
    except KeyboardInterrupt:
        SystemExit("KeyboardInterrupt: Exiting the script.")
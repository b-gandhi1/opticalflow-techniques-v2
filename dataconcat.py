import pandas as pd
import glob
import torch

def concatenate_csv_files(gnd_path, imu_path, mcp_path, skip_rows=20):
    gnd_csv_files = glob.glob(gnd_path)
    data_frames_gnd = [pd.read_csv(f, usecols=['roll_x','pitch_y','yaw_z']) for f in gnd_csv_files]
    imu_csv_files = glob.glob(imu_path)
    data_frames_imu = [pd.read_csv(f, usecols=['Pressure (kPa)','IMU X','IMU Y','IMU Z']) for f in imu_csv_files]
    mcp_csv_files = glob.glob(mcp_path)
    data_frames_mcp = [pd.read_csv(f, usecols=['x_vals','y_vals','z_vals']) for f in mcp_csv_files]
    
    concatenated_mcp = pd.concat(data_frames_mcp, ignore_index=True, axis='index')
    # print(concatenated_mcp.shape)
    
    concatenated_gnd = pd.concat(data_frames_imu, ignore_index=True, axis='index')
    # print(concatenated_imu.shape)
    
    concatenated_imu = pd.concat(data_frames_gnd, ignore_index=True, axis='index')
    
    concatenated_df = pd.concat([concatenated_gnd, concatenated_imu, concatenated_mcp], ignore_index=False, axis='columns')
    # print(concatenated_df.shape)
    
    concatenated_df.columns = ['Franka Rx','Franka Ry','Franka Rz','Pressure (kPa)','IMU Rx','IMU Ry','IMU Rz','LKx','LKy','BrtZ']
    
    # # Pair and concatenate each pair of data frames
    # concatenated_pairs = [pd.concat([imu_df, mcp_df], ignore_index=True) for imu_df, mcp_df in zip(data_frames_imu, data_frames_mcp)]
    
    # # Concatenate all pairs into a single data frame
    # concatenated_df = pd.concat(concatenated_pairs, ignore_index=True)
    
    return concatenated_df

def concatenate_tensor_files(path_pattern, skip_entries=15):
    tensor_files = glob.glob(path_pattern)
    tensors = [torch.load(f)[skip_entries:] for f in tensor_files]
    
    # print a sample
    # for tensor in tensors:
    #     print('size: ', tensor.size())
    #     print(tensor[0])
        
    concatenated_tensor = torch.cat(tensors, dim=0)
    return concatenated_tensor

def main(pitchroll):
    franka_csv_path = 'imu-fusion-data/LK_'+pitchroll+'2/*euler_gnd*.csv' # gnd data
    imu_csv_path = 'imu-fusion-data/LK_'+pitchroll+'2/fibrescope*.csv' # imu data + pressure sensor data
    mcp_csv_path = 'imu-fusion-data/LK_'+pitchroll+'2/imu-fusion-outputs*.csv' # mcp data
    tensorX_path_pattern = 'imu-fusion-data/LK_'+pitchroll+'2/tensor_x*.pt' # tensor_x data
    tensorY_path_pattern = 'imu-fusion-data/LK_'+pitchroll+'2/tensor_y*.pt' # tensor_y data
    
    concatenated_csv = concatenate_csv_files(gnd_path = franka_csv_path, imu_path=imu_csv_path, mcp_path=mcp_csv_path)
    concatenated_tensor_x = concatenate_tensor_files(tensorX_path_pattern)
    concatenated_tensor_y = concatenate_tensor_files(tensorY_path_pattern)
    
    concatenated_csv.to_csv('imu-fusion-data/pitchroll_concat2/'+pitchroll+'_concatenated_data.csv', index=False)
    torch.save(concatenated_tensor_x, 'imu-fusion-data/pitchroll_concat2/'+pitchroll+'_concatenated_tensor_x.pt')
    torch.save(concatenated_tensor_y, 'imu-fusion-data/pitchroll_concat2/'+pitchroll+'_concatenated_tensor_y.pt')

if __name__ == "__main__":
    
    main(pitchroll="pitch")
    main(pitchroll="roll")
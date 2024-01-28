import numpy as np
import pandas as pd
# import math
import pickle
import matplotlib.pyplot as plt

# AIM: Link the OF and BD outputs to the ground truth in Euler format, and find a relationship between the two. 


# load pickle data from outputs folder 

def data_load_adapt(data_var):
    max_len = max(len(d) for d in data_var)
    padded = [np.pad(d, (0, max_len - len(d)), mode='constant') for d in data_var]
    data_out = np.asarray(padded)
    # data_out = data_var
    return data_out

# pressure plot - control vars. 
def plot_pressure(web_pressures, fib_pressures): 
    web_kpa = web_pressures.iloc[1:,0]
    web_pump_state = web_pressures.iloc[1:,1]
    fib_kpa = fib_pressures.iloc[1:,0]
    fib_pump_state = fib_pressures.iloc[1:,1]
    
    plt.figure('Pressure and pump state')
    time_ax_web = np.linspace(0,60,len(web_kpa))
    time_ax_fib = np.linspace(0,60,len(fib_kpa))
    plt.subplot(211)
    plt.plot(time_ax_web,web_kpa,label = 'webcam')
    plt.plot(time_ax_fib,fib_kpa,label = 'fibrescope')
    plt.ylabel('Pressure (kPa)')
    plt.xlabel('Time (s)')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(time_ax_web,web_pump_state,label = 'webcam')
    plt.plot(time_ax_fib,fib_pump_state,label = 'fibrescope')
    plt.ylabel('Pump state (Raw, ASCII)')
    plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.legend()
    
    # plt.savefig('result_figs/controlVarsSample2.svg',format = 'svg')
    plt.show()
    
def linkingIO(gnd_truth_euler, fib_web_dat):
    # get max displacements from fib_web_dat
    # mag = fib_web_dat['magnitude']
    # ang = fib_web_dat['angle']
    test = [data['x_val'] for data in fib_web_dat]
    print('shape of 3d x_val : ',len(test[0]),len(test[1]),len(test[2]))
    x_val = np.max(data_load_adapt([data['x_val'] for data in fib_web_dat]),axis=1)
    print("x_val 1d shape: -----", np.shape(x_val))
    print(x_val)
    y_val = np.max(data_load_adapt([data['y_val'] for data in fib_web_dat]),axis=1)
    z_val = np.asarray([data['z_val'] for data in fib_web_dat],dtype=float) # already 1D
    timestamps = [data['timestamp'] for data in fib_web_dat] # already 1D
    # print('Shape displacements BD: ',np.shape(mag))
    # print('XYZ shapes: ', np.shape(x_val),np.shape(y_val),np.shape(z_val))
    
    euler_x = gnd_truth_euler.iloc[1:,0]
    euler_y = gnd_truth_euler.iloc[1:,1]
    euler_z = gnd_truth_euler.iloc[1:,2]
        
    # plot against gnd_truth_euler
    plt.subplot(311)
    plt.scatter(euler_x,x_val)
    plt.xlabel('Ground truth - Rot(x)')
    plt.ylabel('X Value (from frame)')
    plt.ylim([600,650])
    
    plt.subplot(312)
    plt.scatter(euler_y,y_val)
    plt.xlabel('Ground truth - Rot(y)')
    plt.ylabel('Y Value (from frame)')
    
    plt.subplot(313)
    plt.scatter(euler_z,z_val)
    plt.xlabel('Ground truth - Rot(z)')
    plt.ylabel('Z Value (from frame)')
    
    plt.tight_layout()    

# def linkingIO_OF(gnd_truth_euler, fib_web_dat):
#     # get max displacements from fib_web_dat
#     # mag = fib_web_dat['magnitude']
#     # ang = fib_web_dat['angle']
#     x_val = fib_web_dat['x_val']
#     y_val = fib_web_dat['y_val']
#     z_val = fib_web_dat['z_val']
#     timestamps_OF = fib_web_dat['timestamp']
#     # print('Shape magnituge OF: ',np.shape(mag))
#     # print('Shape angle OF: ',np.shape(ang))
#     print('XYZ shapes: ', np.shape(x_val),np.shape(y_val),np.shape(z_val))

    
#     euler_x = gnd_truth_euler[:,0]
#     euler_y = gnd_truth_euler[:,1]
#     euler_z = gnd_truth_euler[:,2]
    
#     # plot against gnd_truth_euler
    
#     ...
    
def main():
    # load ground truth data from franka, csv file 
    
    fib_df1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope1-20-Nov-2023--14-06-58.csv', delimiter=',')
    fib_gnd_truth_df1 = fib_df1.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions
    fib_df_euler1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fib1euler.csv', delimiter=',',header=None,usecols=[1,2,3])
    
    fib_df2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope2-20-Nov-2023--14-09-23.csv', delimiter=',')
    fib_gnd_truth_df2 = fib_df2.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions
    fib_df_euler2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fib2euler.csv', delimiter=',',header=None,usecols=[1,2,3])
    
    web_df1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/webcam1-20-Nov-2023--15-56-11.csv', delimiter=',')
    web_gnd_truth_df1 = web_df1.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions
    web_df_euler1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/web1euler.csv', delimiter=',',header=None,usecols=[1,2,3])
    
    web_df2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/webcam2-20-Nov-2023--15-59-11.csv', delimiter=',')
    web_gnd_truth_df2 = web_df2.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions
    web_df_euler2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/web2euler.csv', delimiter=',',header=None,usecols=[1,2,3])
    
    # pressure in pillow (kPa) and pump state - plot this for a control measure. 
    fib_pressures2 = fib_df2.iloc[1:,3:4+1]
    web_pressures2 = web_df2.iloc[1:,3:4+1]
    
    # plot_pressure(web_pressures1,fib_pressures1) 
    # plt.title('Sample 1')
    # plot_pressure(web_pressures2,fib_pressures2) # plots control variables in pillow. 
    # plt.title('Sample 2')
    
    # load data from cams
    web_bd_gray1_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_gray_web1_2024-01-23_13-27-23.pkl", "rb"))
    web_bd_gray2_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_gray_web2_2024-01-23_13-28-18.pkl", "rb"))
    fib_bd_gray1_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_gray_fib1_2024-01-23_12-54-44.pkl", "rb"))  
    fib_bd_gray2_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_gray_fib2_2024-01-23_12-56-19.pkl", "rb"))
    
    web_bd_bin1_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_binary_web1_2024-01-23_13-29-48.pkl", "rb"))
    web_bd_bin2_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_binary_web2_2024-01-23_13-30-45.pkl", "rb"))
    fib_bd_bin1_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_binary_fib1_2024-01-23_13-02-54.pkl", "rb"))
    fib_bd_bin2_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_binary_fib2_2024-01-23_13-08-41.pkl", "rb"))
    
    web_lk_gray1_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_gray_web1_2024-01-23_13-25-03.pkl", "rb"))
    web_lk_gray2_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_gray_web2_2024-01-23_13-25-58.pkl", "rb"))
    fib_lk_gray1_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_gray_fib1_2024-01-23_13-13-22.pkl", "rb"))
    fib_lk_gray2_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_gray_fib2_2024-01-23_13-14-09.pkl", "rb"))
    
    web_lk_bin1_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_binary_web1_2024-01-23_13-18-06.pkl", "rb"))
    web_lk_bin2_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_binary_web2_2024-01-23_13-19-24.pkl", "rb"))
    fib_lk_bin1_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_binary_fib1_2024-01-23_13-10-26.pkl", "rb"))
    fib_lk_bin2_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_binary_fib2_2024-01-23_13-11-41.pkl", "rb"))
    
    # adapting datasets for useability:
    # web_bd_gray1, fib_bd_gray1, web_bd_gray2, fib_bd_gray2 = data_load_adapt(web_bd_gray1_raw), data_load_adapt(fib_bd_gray1_raw), data_load_adapt(web_bd_gray2_raw), data_load_adapt(fib_bd_gray2_raw)
    # web_bd_bin1, fib_bd_bin1, web_bd_bin2, fib_bd_bin2 = data_load_adapt(web_bd_bin1_raw), data_load_adapt(fib_bd_bin1_raw), data_load_adapt(web_bd_bin2_raw), data_load_adapt(fib_bd_bin2_raw)
    # web_lk_gray1, fib_lk_gray1, web_lk_gray2, fib_lk_gray2 = data_load_adapt(web_lk_gray1_raw), data_load_adapt(fib_lk_gray1_raw), data_load_adapt(web_lk_gray2_raw), data_load_adapt(fib_lk_gray2_raw)
    # web_lk_bin1, fib_lk_bin1, web_lk_bin2, fib_lk_bin2 = data_load_adapt(web_lk_bin1_raw), data_load_adapt(fib_lk_bin1_raw), data_load_adapt(web_lk_bin2_raw), data_load_adapt(fib_lk_bin2_raw)
    
    # outputs for linking IO plots: 
    # plt.ion()
    # webcam: 
    plt.figure(1)
    linkingIO(web_df_euler1,web_bd_gray1_raw) # for BD_gray
    plt.suptitle('Sample 1: Grayscale Webcam + Blob Detection')
    plt.figure(2)
    linkingIO(web_df_euler2,web_bd_gray2_raw)
    plt.suptitle('Sample 2: Grayscale Webcam + Blob Detection')
    
    plt.figure(3)
    linkingIO(web_df_euler1,web_bd_bin1_raw) # for BD_binary
    plt.suptitle('Sample 1: Binary Webcam + Blob Detection')
    plt.figure(4)
    linkingIO(web_df_euler2,web_bd_bin2_raw)
    plt.suptitle('Sample 2: Binary Webcam + Blob Detection')
    
    plt.figure(5)
    linkingIO(web_df_euler1,web_lk_gray1_raw) # for LK_gray
    plt.suptitle('Sample 1: Grayscale Webcam + Optical Flow')
    plt.figure(6)
    linkingIO(web_df_euler2,web_lk_gray2_raw)
    plt.suptitle('Sample 2: Grayscale Webcam + Optical Flow')
    
    plt.figure(7)
    linkingIO(web_df_euler1,web_lk_bin1_raw) # for LK_binary
    plt.suptitle('Sample 1: Binary Webcam + Optical Flow')
    plt.figure(8)
    linkingIO(web_df_euler2,web_lk_bin2_raw)
    plt.suptitle('Sample 2: Binary Webcam + Optical Flow')
    
    # fibrescope:
    plt.figure(9) 
    linkingIO(fib_df_euler1,fib_bd_gray1_raw) # for BD_gray
    plt.suptitle('Sample 1: Grayscale Fibrescope + Blob Detection')
    plt.figure(10)
    linkingIO(fib_df_euler2,fib_bd_gray2_raw)
    plt.suptitle('Sample 2: Grayscale Fibrescope + Blob Detection')
    
    plt.figure(11)
    linkingIO(fib_df_euler1,fib_bd_bin1_raw) # for BD_binary
    plt.suptitle('Sample 1: Binary Fibrescope + Blob Detection')
    plt.figure(12)
    linkingIO(fib_df_euler2,fib_bd_bin2_raw)
    plt.suptitle('Sample 2: Binary Fibrescope + Blob Detection')
    
    plt.figure(13)
    linkingIO(fib_df_euler1,fib_lk_gray1_raw) # for LK_gray
    plt.suptitle('Sample 1: Grayscale Fibrescope + Optical Flow')
    plt.figure(14)
    linkingIO(fib_df_euler2,fib_lk_gray2_raw)
    plt.suptitle('Sample 2: Grayscale Fibrescope + Optical Flow')
    
    plt.figure(15)
    linkingIO(fib_df_euler1,fib_lk_bin1_raw) # for LK_binary
    plt.suptitle('Sample 1: Binary Fibrescope + Optical Flow')
    plt.figure(16)
    linkingIO(fib_df_euler2,fib_lk_bin2_raw)
    plt.suptitle('Sample 2: Binary Fibrescope + Optical Flow')
    
    plt.show()
if __name__ == "__main__":
    main()
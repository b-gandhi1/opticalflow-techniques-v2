import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt

# AIM: Link the OF and BD outputs to the ground truth in Euler format, and find a relationship between the two. 


# load pickle data from outputs folder 

def data_load_adapt(data_var):
    max_len = max(len(d['data']) for d in data_var)
    padded = [np.pad(d['data'], (0, max_len - len(d['data'])), mode='constant') for d in data_var]
    data_out = np.asarray(padded)
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
    
def linkingIO_BD(gnd_truth_euler, fib_web_dat):
    # get max displacements from fib_web_dat
    displacements = fib_web_dat['data']
    x_val = fib_web_dat['x_val']
    y_val = fib_web_dat['y_val']
    timestamps_BD = fib_web_dat['timestamp']
    print('Shape displacements BD: ',np.shape(displacements))
    
    euler_x = gnd_truth_euler[:,0]
    euler_y = gnd_truth_euler[:,1]
    euler_z = gnd_truth_euler[:,2]
    
    # find max in 
    
    # plot against gnd_truth_euler
    plt.figure()
    plt.plot()
    ... 

def linkingIO_OF(gnd_truth_euler, fib_web_dat):
    # get max displacements from fib_web_dat
    mag = fib_web_dat['data'][...,0]
    ang = fib_web_dat['data'][...,1]
    timestamps_OF = fib_web_dat['timestamp']
    print('Shape magnituge OF: ',np.shape(mag))
    print('Shape angle OF: ',np.shape(ang))
    
    euler_x = gnd_truth_euler[:,0]
    euler_y = gnd_truth_euler[:,1]
    euler_z = gnd_truth_euler[:,2]
    
    # plot against gnd_truth_euler
    
    ...
    
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
    web_bd_gray1_raw = pickle.load(open("OF_outputs/BD_gray_web1_2023-12-06_16-57-23.pkl", "rb"))
    web_bd_gray2_raw = pickle.load(open("OF_outputs/BD_gray_web2_2023-12-06_16-58-18.pkl", "rb"))
    fib_bd_gray1_raw = pickle.load(open("OF_outputs/BD_gray_fib1_2023-12-07_12-30-19.pkl", "rb"))  
    fib_bd_gray2_raw = pickle.load(open("OF_outputs/BD_gray_fib2_2023-12-07_12-31-11.pkl", "rb"))
    
    web_bd_bin1_raw = pickle.load(open("OF_outputs/BD_binary_web1_2023-12-06_16-49-54.pkl", "rb"))
    web_bd_bin2_raw = pickle.load(open("OF_outputs/BD_binary_web2_2023-12-06_16-52-02.pkl", "rb"))
    fib_bd_bin1_raw = pickle.load(open("OF_outputs/BD_binary_fib1_2023-12-06_17-47-48.pkl", "rb"))
    fib_bd_bin2_raw = pickle.load(open("OF_outputs/BD_binary_fib2_2023-12-06_17-48-47.pkl", "rb"))
    
    web_lk_gray1_raw = pickle.load(open("OF_outputs/LK_gray_web1_2023-12-04_17-02-48.pkl", "rb"))
    web_lk_gray2_raw = pickle.load(open("OF_outputs/LK_gray_web2_2023-12-04_17-04-32.pkl", "rb"))
    fib_lk_gray1_raw = pickle.load(open("OF_outputs/LK_bright_fib1_2023-11-29_16-44-01.pkl", "rb"))
    fib_lk_gray2_raw = pickle.load(open("OF_outputs/LK_bright_fib2_2023-11-29_16-44-52.pkl", "rb"))
    
    web_lk_bin1_raw = pickle.load(open("OF_outputs/LK_binary_web1_2023-12-06_17-56-36.pkl", "rb"))
    web_lk_bin2_raw = pickle.load(open("OF_outputs/LK_binary_web2_2023-12-06_17-57-30.pkl", "rb"))
    fib_lk_bin1_raw = pickle.load(open("OF_outputs/LK_binary_fib1_2023-11-29_16-37-46.pkl", "rb"))
    fib_lk_bin2_raw = pickle.load(open("OF_outputs/LK_binary_fib2_2023-11-29_16-39-21.pkl", "rb"))
    
    # adapting datasets for useability:
    web_bd_gray1, fib_bd_gray1, web_bd_gray2, fib_bd_gray2 = data_load_adapt(web_bd_gray1_raw), data_load_adapt(fib_bd_gray1_raw), data_load_adapt(web_bd_gray2_raw), data_load_adapt(fib_bd_gray2_raw)
    web_bd_bin1, fib_bd_bin1, web_bd_bin2, fib_bd_bin2 = data_load_adapt(web_bd_bin1_raw), data_load_adapt(fib_bd_bin1_raw), data_load_adapt(web_bd_bin2_raw), data_load_adapt(fib_bd_bin2_raw)
    web_lk_gray1, fib_lk_gray1, web_lk_gray2, fib_lk_gray2 = data_load_adapt(web_lk_gray1_raw), data_load_adapt(fib_lk_gray1_raw), data_load_adapt(web_lk_gray2_raw), data_load_adapt(fib_lk_gray2_raw)
    web_lk_bin1, fib_lk_bin1, web_lk_bin2, fib_lk_bin2 = data_load_adapt(web_lk_bin1_raw), data_load_adapt(fib_lk_bin1_raw), data_load_adapt(web_lk_bin2_raw), data_load_adapt(fib_lk_bin2_raw)
    
    # outputs for linking IO plots: 
    # webcam: 
    linkingIO_BD(web_df_euler1,web_bd_gray1) # for BD_gray
    plt.title('Sample 1: Grayscale Webcam + Blob Detection')
    linkingIO_BD(web_df_euler2,web_bd_gray2)
    plt.title('Sample 2: Grayscale Webcam + Blob Detection')
    
    linkingIO_BD(web_df_euler1,web_bd_bin1) # for BD_binary
    plt.title('Sample 1: Binary Webcam + Blob Detection')
    linkingIO_BD(web_df_euler2,web_bd_bin2)
    plt.title('Sample 2: Binary Webcam + Blob Detection')
    
    linkingIO_OF(web_df_euler1,web_lk_gray1) # for LK_gray
    plt.title('Sample 1: Grayscale Webcam + Optical Flow')
    linkingIO_OF(web_df_euler2,web_lk_gray2)
    plt.title('Sample 2: Grayscale Webcam + Optical Flow')
    
    linkingIO_OF(web_df_euler1,web_lk_bin1) # for LK_binary
    plt.title('Sample 1: Binary Webcam + Optical Flow')
    linkingIO_OF(web_df_euler2,web_lk_bin2)
    plt.title('Sample 2: Binary Webcam + Optical Flow')
    
    # fibrescope: 
    linkingIO_BD(fib_df_euler1,fib_bd_gray1) # for BD_gray
    plt.title('Sample 1: Grayscale Fibrescope + Blob Detection')
    linkingIO_BD(fib_df_euler2,fib_bd_gray2)
    plt.title('Sample 2: Grayscale Fibrescope + Blob Detection')
    
    linkingIO_BD(fib_df_euler1,fib_bd_bin1) # for BD_binary
    plt.title('Sample 1: Binary Fibrescope + Blob Detection')
    linkingIO_BD(fib_df_euler2,fib_bd_bin2)
    plt.title('Sample 2: Binary Fibrescope + Blob Detection')
    
    linkingIO_OF(fib_df_euler1,fib_lk_gray1) # for LK_gray
    plt.title('Sample 1: Grayscale Fibrescope + Optical Flow')
    linkingIO_OF(fib_df_euler2,fib_lk_gray2)
    plt.title('Sample 2: Grayscale Fibrescope + Optical Flow')
    
    linkingIO_OF(fib_df_euler1,fib_lk_bin1) # for LK_binary
    plt.title('Sample 1: Binary Fibrescope + Optical Flow')
    linkingIO_OF(fib_df_euler2,fib_lk_bin2)
    plt.title('Sample 2: Binary Fibrescope + Optical Flow')
    
if __name__ == "__main__":
    main()
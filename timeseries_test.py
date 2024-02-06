# this file is to test the time series trends from the blob detection and optical flow algorithms

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def load_data():
    # load data from cams
    # web_bd_gray1_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_gray_web1_2024-01-23_13-27-23.pkl", "rb"))
    # web_bd_gray2_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_gray_web2_2024-01-23_13-28-18.pkl", "rb"))
    # fib_bd_gray1_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_gray_fib1_2024-01-23_12-54-44.pkl", "rb"))  
    # fib_bd_gray2_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_gray_fib2_2024-01-23_12-56-19.pkl", "rb"))
    
    # web_bd_bin1_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_binary_web1_2024-01-23_13-29-48.pkl", "rb"))
    # web_bd_bin2_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_binary_web2_2024-01-23_13-30-45.pkl", "rb"))
    # fib_bd_bin1_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_binary_fib1_2024-01-23_13-02-54.pkl", "rb"))
    # fib_bd_bin2_raw = pickle.load(open("OF_outputs/data2_jan2023/BD_binary_fib2_2024-01-23_13-08-41.pkl", "rb"))
    
    # web_lk_gray1_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_gray_web1_2024-01-23_13-25-03.pkl", "rb"))
    # web_lk_gray2_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_gray_web2_2024-01-23_13-25-58.pkl", "rb"))
    # fib_lk_gray1_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_gray_fib1_2024-01-23_13-13-22.pkl", "rb"))
    # fib_lk_gray2_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_gray_fib2_2024-01-23_13-14-09.pkl", "rb"))
    
    # web_lk_bin1_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_binary_web1_2024-01-23_13-18-06.pkl", "rb"))
    # web_lk_bin2_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_binary_web2_2024-01-23_13-19-24.pkl", "rb"))
    # fib_lk_bin1_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_binary_fib1_2024-01-23_13-10-26.pkl", "rb"))
    # fib_lk_bin2_raw = pickle.load(open("OF_outputs/data2_jan2023/LK_binary_fib2_2024-01-23_13-11-41.pkl", "rb"))
    
    # csv data from arm: 
    fib_df_euler1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fib1euler.csv', delimiter=',',header=None)
    
    fib_df_euler2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fib2euler.csv', delimiter=',',header=None)
    
    web_df_euler1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/web1euler.csv', delimiter=',',header=None)
    # print(np.shape(web_df_euler1))
    
    web_df_euler2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/web2euler.csv', delimiter=',',header=None)
    
    data1 = fib_df_euler1
    return data1

# def data_load_adapt(data_var):
#     max_len = max(len(d) for d in data_var)
#     padded = [np.pad(d, (0, max_len - len(d)), mode='constant') for d in data_var]
#     data_out = np.asarray(padded)
#     # data_out = data_var
#     return data_out

def oneD_plots(timeseries):
    plt.figure()
    plt.plot(range(len(timeseries)),timeseries)
    plt.tight_layout()
    plt.show()

def main():
    data = load_data()
    # x_mat = np.asarray([data['x_val_1d'] for data in data],dtype=float) # for pickle files
    x,y,z = 1,2,3
    # w,x,y,z = 0,1,2,3
    # w_vals = data.iloc[1:,w]
    x_vals = data.iloc[1:,x]
    x_vals = [float(x) for x in x_vals]
    y_vals = data.iloc[1:,y]
    y_vals = [float(y) for y in y_vals]
    z_vals = data.iloc[1:,z]
    z_vals = [float(z) for z in z_vals]

    # oneD_plots(x_vals)
    
    plt.figure()
    # plt.plot(range(len(w_vals)),w_vals)
    plt.plot(range(len(x_vals)),(x_vals))
    plt.plot(range(len(y_vals)),y_vals)
    plt.plot(range(len(z_vals)),z_vals)
    plt.legend(['roll_x','pitch_y','yaw_z'],loc='upper right')
    # plt.plot(range(len(z_vals)),w_z'])
    # plt.legend(['w','x','y','z'])
    # plt.ylim((-1,2*np.pi))
    # plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
    
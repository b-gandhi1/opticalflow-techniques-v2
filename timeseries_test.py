# this file is to test the time series trends from the blob detection and optical flow algorithms

import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_data():
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
    
    # trials: 
    data1 = pickle.load(open("OF_outputs/LK_binary_web_2024-01-28_21-47-54.pkl","rb"))
    data2 = pickle.load(open("OF_outputs/BD_binary_web_2024-01-28_21-24-08.pkl","rb"))
    
    return data1

def data_load_adapt(data_var):
    max_len = max(len(d) for d in data_var)
    padded = [np.pad(d, (0, max_len - len(d)), mode='constant') for d in data_var]
    data_out = np.asarray(padded)
    # data_out = data_var
    return data_out

def oneD_plots(timeseries):
    plt.figure()
    plt.plot(range(len(timeseries)),timeseries)
    plt.tight_layout()
    # plt.show()

def main():
    data = load_data()
    # print('shape of data: ', len(data[0]),len(data[1]),len(data[2]),len(data[3]),len(data[4]),len(data[5]),len(data[6]),len(data[7]),len(data[8]),len(data[9]),len(data[10]),len(data[11]),len(data[12]),len(data[13]),len(data[14]),len(data[15]),len(data[16]),len(data[17]),len(data[18]),len(data[19]),len(data[20]),len(data[21]))
    x_mat = np.asarray([data['x_val_1d'] for data in data],dtype=float)
    # print('len of x_mat: ', len(x_mat[0]))
    print('shape of x_mat: ', np.shape(x_mat))
    # x_mat_padded = data_load_adapt(x_mat)
    # print('shape of x_1d_padded: ', np.shape(x_mat_padded))
    # x_1d = np.median(x_mat_padded,axis=1)
    # print('shape of x_1d: ', np.shape(x_1d))
    # oneD_plots(x_mat)
    
# if __name__ == "__main__":
#     main()
    
import pickle
import pandas as pd
import numpy as np
import sys

path = sys.argv[1]
savefile = sys.argv[2]
# load data from cams

pickle_load = pickle.load(open(path,"rb"))

# web_bd_gray1_raw = pickle.load(open("OF_outputs/data4_feb2024/BD_gray_web2_2024-02-06_17-15-22.pkl", "rb"))
# web_bd_gray2_raw = pickle.load(open("OF_outputs/data4_feb2024/BD_gray_web2_2024-02-06_17-15-22.pkl", "rb"))
# fib_bd_gray1_raw = pickle.load(open("OF_outputs/data4_feb2024/BD_gray_fib2_2024-02-06_17-36-31.pkl", "rb"))  
# fib_bd_gray2_raw = pickle.load(open("OF_outputs/data4_feb2024/BD_gray_fib2_2024-02-06_17-36-31.pkl", "rb"))

# web_bd_bin1_raw = pickle.load(open("OF_outputs/data4_feb2024/BD_binary_web2_2024-02-06_18-28-22.pkl", "rb"))
# web_bd_bin2_raw = pickle.load(open("OF_outputs/data4_feb2024/BD_binary_web2_2024-02-06_18-28-22.pkl", "rb"))
# fib_bd_bin1_raw = pickle.load(open("OF_outputs/data4_feb2024/BD_binary_fib2_2024-02-06_17-42-17.pkl", "rb"))
# fib_bd_bin2_raw = pickle.load(open("OF_outputs/data4_feb2024/BD_binary_fib2_2024-02-06_17-42-17.pkl", "rb"))

# web_lk_gray1_raw = pickle.load(open("OF_outputs/data4_feb2024/LK_gray_web2_2024-02-06_17-15-22.pkl", "rb"))
# web_lk_gray2_raw = pickle.load(open("OF_outputs/data4_feb2024/LK_gray_web2_2024-02-06_17-15-22.pkl", "rb"))
# fib_lk_gray1_raw = pickle.load(open("OF_outputs/data4_feb2024/LK_gray_fib2_2024-02-06_17-36-31.pkl", "rb"))
# fib_lk_gray2_raw = pickle.load(open("OF_outputs/data4_feb2024/LK_gray_fib2_2024-02-06_17-36-31.pkl", "rb"))

# web_lk_bin1_raw = pickle.load(open("OF_outputs/data4_feb2024/LK_binary_web2_2024-02-06_18-28-22.pkl", "rb"))
# web_lk_bin2_raw = pickle.load(open("OF_outputs/data4_feb2024/LK_binary_web2_2024-02-06_18-28-22.pkl", "rb"))
# fib_lk_bin1_raw = pickle.load(open("OF_outputs/data4_feb2024/LK_binary_fib2_2024-02-06_17-42-17.pkl", "rb"))
# fib_lk_bin2_raw = pickle.load(open("OF_outputs/data4_feb2024/LK_binary_fib2_2024-02-06_17-42-17.pkl", "rb"))

# extract 1D xyz vars

x_vals = np.asarray([data['x_val_1d'] for data in pickle_load],dtype=float)
y_vals = np.asarray([data['y_val_1d'] for data in pickle_load],dtype=float)
z_vals = np.asarray([data['z_val'] for data in pickle_load],dtype=float)

# web_bd_gray_x = np.asarray([data['x_val_1d'] for data in web_bd_gray1_raw],dtype=float)
# web_bd_gray_y = np.asarray([data['y_val_1d'] for data in web_bd_gray1_raw],dtype=float)
# web_bd_bin_x = np.asarray([data['x_val_1d'] for data in web_bd_bin1_raw],dtype=float)
# web_bd_bin_y = np.asarray([data['y_val_1d'] for data in web_bd_bin1_raw],dtype=float)
# web_lk_gray_x = np.asarray([data['x_val_1d'] for data in web_lk_gray1_raw],dtype=float)
# web_lk_gray_y = np.asarray([data['y_val_1d'] for data in web_lk_gray1_raw],dtype=float)
# web_lk_bin_x = np.asarray([data['x_val_1d'] for data in web_lk_bin1_raw],dtype=float)
# web_lk_bin_y = np.asarray([data['y_val_1d'] for data in web_lk_bin1_raw],dtype=float)
# fib_bd_gray_x = np.asarray([data['x_val_1d'] for data in fib_bd_gray1_raw],dtype=float)
# fib_bd_gray_y = np.asarray([data['y_val_1d'] for data in fib_bd_gray1_raw],dtype=float)
# fib_bd_bin_x = np.asarray([data['x_val_1d'] for data in fib_bd_bin1_raw],dtype=float)
# fib_bd_bin_y = np.asarray([data['y_val_1d'] for data in fib_bd_bin1_raw],dtype=float)
# fib_lk_gray_x = np.asarray([data['x_val_1d'] for data in fib_lk_gray1_raw],dtype=float)
# fib_lk_gray_y = np.asarray([data['y_val_1d'] for data in fib_lk_gray1_raw],dtype=float)
# fib_lk_bin_x = np.asarray([data['x_val_1d'] for data in fib_lk_bin1_raw],dtype=float)
# fib_lk_bin_y = np.asarray([data['y_val_1d'] for data in fib_lk_bin1_raw],dtype=float)

# 1d z vars 
# web_bd_gray_z = np.asarray([data['z_val'] for data in web_bd_gray1_raw],dtype=float)
# web_bd_bin_z = np.asarray([data['z_val'] for data in web_bd_bin1_raw],dtype=float)
# fib_bd_gray_z = np.asarray([data['z_val'] for data in fib_bd_gray1_raw],dtype=float)
# fib_bd_bin_z = np.asarray([data['z_val'] for data in fib_bd_bin1_raw],dtype=float)

# put it all on a csv file and save it

pd = pd.DataFrame({'x_vals':x_vals,'y_vals':y_vals,'z_vals':z_vals},dtype=float)
pd.to_csv('imu-fusion-outputs_LK_Zavg'+savefile+'.csv',header=True)

print("--- END ---")
# fib_pd = pd.DataFrame({'bd_gray_x':fib_bd_gray_x,'bd_gray_y':fib_bd_gray_y,'lk_gray_x':fib_lk_gray_x,'lk_gray_y':fib_lk_gray_y,'z_brightness_gray':fib_bd_gray_z,'z_brightness_bin':fib_bd_bin_z},dtype=float)
# web_pd = pd.DataFrame({'bd_gray_x':web_bd_gray_x,'bd_gray_y':web_bd_gray_y,'lk_gray_x':web_lk_gray_x,'lk_gray_y':web_lk_gray_y,'z_brightness_gray':web_bd_gray_z,'z_brightness_bin':web_bd_bin_z},dtype=float)

# fib_pd.to_csv('OF_outputs/fib_pd.csv',header=True)
# print('fib_pd done')
# web_pd.to_csv('OF_outputs/web_pd.csv',header=True)
# print('web_pd done')

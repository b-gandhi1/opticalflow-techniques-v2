import pickle
import pandas as pd
import numpy as np

# load data from cams
web_bd_gray1_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_gray_web1_2024-01-30_11-35-35.pkl", "rb"))
# web_bd_gray2_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_gray_web2_2024-01-30_11-36-49.pkl", "rb"))
fib_bd_gray1_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_gray_fib1_2024-01-30_11-14-19.pkl", "rb"))  
# fib_bd_gray2_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_gray_fib2_2024-01-30_11-15-07.pkl", "rb"))

web_bd_bin1_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_binary_web1_2024-01-30_11-22-31.pkl", "rb"))
# web_bd_bin2_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_binary_web2_2024-01-30_11-25-56.pkl", "rb"))
fib_bd_bin1_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_binary_fib1_2024-01-30_11-16-48.pkl", "rb"))
# fib_bd_bin2_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_binary_fib2_2024-01-30_11-19-19.pkl", "rb"))

web_lk_gray1_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_gray_web1_2024-01-30_11-35-35.pkl", "rb"))
# web_lk_gray2_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_gray_web2_2024-01-30_11-36-49.pkl", "rb"))
fib_lk_gray1_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_gray_fib1_2024-01-30_11-04-56.pkl", "rb"))
# fib_lk_gray2_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_gray_fib2_2024-01-30_11-10-53.pkl", "rb"))

web_lk_bin1_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_binary_web1_2024-01-30_11-22-31.pkl", "rb"))
# web_lk_bin2_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_binary_web2_2024-01-30_11-25-56.pkl", "rb"))
fib_lk_bin1_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_binary_fib1_2024-01-30_11-17-34.pkl", "rb"))
# fib_lk_bin2_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_binary_fib2_2024-01-30_11-18-24.pkl", "rb"))

# extract 1D xyz vars
web_bd_gray_x = np.asarray([data['x_val_1d'] for data in web_bd_gray1_raw],dtype=float)
web_bd_gray_y = np.asarray([data['y_val_1d'] for data in web_bd_gray1_raw],dtype=float)
web_bd_bin_x = np.asarray([data['x_val_1d'] for data in web_bd_bin1_raw],dtype=float)
web_bd_bin_y = np.asarray([data['y_val_1d'] for data in web_bd_bin1_raw],dtype=float)
web_lk_gray_x = np.asarray([data['x_val_1d'] for data in web_lk_gray1_raw],dtype=float)
web_lk_gray_y = np.asarray([data['y_val_1d'] for data in web_lk_gray1_raw],dtype=float)
web_lk_bin_x = np.asarray([data['x_val_1d'] for data in web_lk_bin1_raw],dtype=float)
web_lk_bin_y = np.asarray([data['y_val_1d'] for data in web_lk_bin1_raw],dtype=float)
fib_bd_gray_x = np.asarray([data['x_val_1d'] for data in fib_bd_gray1_raw],dtype=float)
fib_bd_gray_y = np.asarray([data['y_val_1d'] for data in fib_bd_gray1_raw],dtype=float)
fib_bd_bin_x = np.asarray([data['x_val_1d'] for data in fib_bd_bin1_raw],dtype=float)
fib_bd_bin_y = np.asarray([data['y_val_1d'] for data in fib_bd_bin1_raw],dtype=float)
fib_lk_gray_x = np.asarray([data['x_val_1d'] for data in fib_lk_gray1_raw],dtype=float)
fib_lk_gray_y = np.asarray([data['y_val_1d'] for data in fib_lk_gray1_raw],dtype=float)
fib_lk_bin_x = np.asarray([data['x_val_1d'] for data in fib_lk_bin1_raw],dtype=float)
fib_lk_bin_y = np.asarray([data['y_val_1d'] for data in fib_lk_bin1_raw],dtype=float)

# 1d z vars 
web_bd_gray_z = np.asarray([data['z_val'] for data in web_bd_gray1_raw],dtype=float)
web_bd_bin_z = np.asarray([data['z_val'] for data in web_bd_bin1_raw],dtype=float)
fib_bd_gray_z = np.asarray([data['z_val'] for data in fib_bd_gray1_raw],dtype=float)
fib_bd_bin_z = np.asarray([data['z_val'] for data in fib_bd_bin1_raw],dtype=float)

# put it all on a csv file and save it
fib_pd = pd.DataFrame({'bd_gray_x':fib_bd_gray_x,'bd_gray_y':fib_bd_gray_y,'lk_gray_x':fib_lk_gray_x,'lk_gray_y':fib_lk_gray_y,'z_brightness_gray':fib_bd_gray_z,'z_brightness_bin':fib_bd_bin_z})
web_pd = pd.DataFrame({'bd_gray_x':web_bd_gray_x,'bd_gray_y':web_bd_gray_y,'lk_gray_x':web_lk_gray_x,'lk_gray_y':web_lk_gray_y,'z_brightness_gray':web_bd_gray_z,'z_brightness_bin':web_bd_bin_z})

fib_pd.to_csv('OF_outputs/fib_pd.csv')
web_pd.to_csv('OF_outputs/web_pd.csv')
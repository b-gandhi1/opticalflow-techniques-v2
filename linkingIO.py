import numpy as np
import pandas as pd
# import math
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing as prep
import sklearn.metrics as metrics
from scipy.stats import pearsonr, spearmanr, weightedtau, kendalltau

# AIM: Link the OF and BD outputs to the ground truth in Euler format, and find a relationship between the two. 


# load pickle data from outputs folder 

# def data_load_adapt(data_var):
#     max_len = max(len(d) for d in data_var)
#     padded = [np.pad(d, (0, max_len - len(d)), mode='constant') for d in data_var]
#     data_out = np.asarray(padded)
#     # data_out = data_var
#     return data_out

# pressure plot - control vars. 

statistics = []

def normalize_vector(vector):
    min_value = np.min(vector)
    max_value = np.max(vector)
    
    normalized_vector = (vector - min_value) / (max_value - min_value) * 2 - 1
    
    return normalized_vector

def statistics_calc(experimental_data, ground_truth):
    corr_linear, _ = pearsonr(experimental_data, ground_truth) # pearson correlation (linear). Low corr ~= 0. -0.5 < poor corr < 0.5. -1 < corr_range < 1. 
    # variables can be +vely or -vely linearly correlated. 
    corr_nonlin, _ = spearmanr(experimental_data, ground_truth) # spearman correlation (nonlinear). High corr ~= 1. -1 < corr_range < 1. same as pearson.
    r_sq = metrics.r2_score(ground_truth,experimental_data)
    corr_kendalltau = kendalltau(ground_truth,experimental_data)
    corr_weightedtau = weightedtau(ground_truth,experimental_data, rank=True, weigher=None, additive=False)

    # TEST THE COMMANDS ABOVE !!!! ----------
    return corr_linear, corr_nonlin, r_sq, corr_kendalltau[0], corr_kendalltau[1], corr_weightedtau[0]

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
    
def linkingIO(gnd_truth_euler, fib_web_dat, t_z_gnd):
    # get max displacements from fib_web_dat
    # mag = fib_web_dat['magnitude']
    # ang = fib_web_dat['angle']
    # test = [data['x_val_1d'] for data in fib_web_dat]
    # print('shape of 3d x_val : ',len(test[0]),len(test[1]),len(test[2]))
    x_val = np.asarray([data['x_val_1d'] for data in fib_web_dat],dtype=float)
    # print("x_val 1d shape: -----", np.shape(x_val))
    # print('Data type of elements in x_val: ',x_val.dtype)
    y_val = np.asarray([data['y_val_1d'] for data in fib_web_dat],dtype=float)
    z_val = np.asarray([data['z_val'] for data in fib_web_dat],dtype=float) # already 1D
    timestamps = [data['timestamp'] for data in fib_web_dat] # already 1D
    # print('Shape displacements BD: ',np.shape(mag))
    
    # Trim to reduct length to 1/2
    half_len = int(0.5*len(z_val))
    x_val = x_val[0:half_len]
    # x_val_norm = x_val/np.linalg.norm(x_val)
    x_val_norm = normalize_vector(x_val)
    y_val = y_val[0:half_len]
    # y_val_norm = y_val/np.linalg.norm(y_val)
    y_val_norm = normalize_vector(y_val)
    z_val = z_val[0:half_len]
    # z_val_norm = z_val/np.linalg.norm(z_val)
    z_val_norm = normalize_vector(z_val)
    
    # print('XYZ shapes: ', np.shape(x_val),np.shape(y_val),np.shape(z_val))
    
    euler_x = np.asarray(gnd_truth_euler.iloc[0:half_len,0])
    # euler_x_norm = euler_x/np.linalg.norm(euler_x)
    euler_x_norm = normalize_vector(euler_x)
    euler_y = np.asarray(gnd_truth_euler.iloc[0:half_len,1])
    # euler_y_norm = euler_y/np.linalg.norm(euler_y)
    euler_y_norm = normalize_vector(euler_y)
    euler_z = np.asarray(gnd_truth_euler.iloc[0:half_len,2])
    # euler_z_norm = euler_z/np.linalg.norm(euler_z)
    euler_z_norm = normalize_vector(euler_z)

    t_z_gnd = np.asarray(t_z_gnd.iloc[0:half_len])
    # t_z_gnd_norm = t_z_gnd/np.linalg.norm(t_z_gnd)
    t_z_gnd_norm = normalize_vector(t_z_gnd)
    
    # plot against gnd_truth_euler
    plt.subplot(411)
    # plt.scatter(euler_x,x_val)
    # plt.xlabel('Ground truth - Rot(x)')
    # plt.ylabel('X Value (from frame)')
    plt.plot(euler_x_norm)
    plt.plot(x_val_norm)
    # plt.ylim([-2*np.pi,2*np.pi])
    plt.legend(['gnd truth','x_val'],loc='upper right')
    plt.tight_layout()
    
    plt.subplot(412)
    # plt.scatter(euler_y,y_val)
    # plt.xlabel('Ground truth - Rot(y)')
    # plt.ylabel('Y Value (from frame)')
    plt.plot(euler_y_norm)
    plt.plot(y_val_norm)
    # plt.ylim([-2*np.pi,2*np.pi])
    plt.tight_layout()
    
    plt.subplot(413)
    # plt.scatter(euler_z,z_val)
    # plt.xlabel('Ground truth - Rot(z)')
    # plt.ylabel('Z Value (from frame)')
    plt.plot(euler_z_norm)
    plt.plot(z_val_norm)
    # plt.ylim([-2*np.pi,2*np.pi])
    plt.tight_layout()
    
    plt.subplot(414)
    # plt.scatter(t_z_gnd,z_val)
    # plt.xlabel('Ground truth - Trans(z)')
    # plt.ylabel('Z Value (from frame)')
    plt.plot(t_z_gnd_norm)
    plt.plot(z_val_norm)
    # plt.ylim([-2*np.pi,2*np.pi])
    plt.tight_layout()
    
    # tabulate statistics: 
    statistics.append(statistics_calc(x_val,euler_x))
    statistics.append(statistics_calc(y_val,euler_y))
    statistics.append(statistics_calc(z_val,euler_z))
    statistics.append(statistics_calc(z_val,t_z_gnd))
    
    return statistics
def main():
    # load ground truth data from franka, csv file 
    
    # fib_df1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope1-20-Nov-2023--14-06-58.csv', delimiter=',')
    # fib_gnd_truth_df1 = fib_df1.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions
    fib_df_euler1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fib1euler.csv', delimiter=',',header=None,usecols=[1,2,3])
    fib_df_euler1 = fib_df_euler1.iloc[1:901,:].astype(float) # trim to 900 vals
    
    # fib_df2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope2-20-Nov-2023--14-09-23.csv', delimiter=',')
    # fib_gnd_truth_df2 = fib_df2.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions
    fib_df_euler2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fib2euler.csv', delimiter=',',header=None,usecols=[1,2,3])
    fib_df_euler2 = fib_df_euler2.iloc[1:901,:].astype(float) # trim to 900 vals
    
    # web_df1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/webcam1-20-Nov-2023--15-56-11.csv', delimiter=',')
    # web_gnd_truth_df1 = web_df1.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions
    web_df_euler1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/web1euler.csv', delimiter=',',header=None,usecols=[1,2,3])
    web_df_euler1 = web_df_euler1.iloc[1:901,:].astype(float) # trim to 900 vals
    
    # web_df2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/webcam2-20-Nov-2023--15-59-11.csv', delimiter=',')
    # web_gnd_truth_df2 = web_df2.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions
    web_df_euler2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/web2euler.csv', delimiter=',',header=None,usecols=[1,2,3])
    web_df_euler2 = web_df_euler2.iloc[1:901,:].astype(float) # trim to 900 vals
    
    # pressure in pillow (kPa) and pump state - plot this for a control measure. 
    # fib_pressures2 = fib_df2.iloc[1:,3:4+1]
    # web_pressures2 = web_df2.iloc[1:,3:4+1]
    
    # plot_pressure(web_pressures1,fib_pressures1) 
    # plt.title('Sample 1')
    # plot_pressure(web_pressures2,fib_pressures2) # plots control variables in pillow. 
    # plt.title('Sample 2')
    
    # load data from cams
    web_bd_gray1_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_gray_web1_2024-01-30_11-35-35.pkl", "rb"))
    web_bd_gray2_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_gray_web2_2024-01-30_11-36-49.pkl", "rb"))
    fib_bd_gray1_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_gray_fib1_2024-01-30_11-14-19.pkl", "rb"))  
    fib_bd_gray2_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_gray_fib2_2024-01-30_11-15-07.pkl", "rb"))
    
    web_bd_bin1_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_binary_web1_2024-01-30_11-22-31.pkl", "rb"))
    web_bd_bin2_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_binary_web2_2024-01-30_11-25-56.pkl", "rb"))
    fib_bd_bin1_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_binary_fib1_2024-01-30_11-16-48.pkl", "rb"))
    fib_bd_bin2_raw = pickle.load(open("OF_outputs/data3_jan2023/BD_binary_fib2_2024-01-30_11-19-19.pkl", "rb"))
    
    web_lk_gray1_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_gray_web1_2024-01-30_11-35-35.pkl", "rb"))
    web_lk_gray2_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_gray_web2_2024-01-30_11-36-49.pkl", "rb"))
    fib_lk_gray1_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_gray_fib1_2024-01-30_11-04-56.pkl", "rb"))
    fib_lk_gray2_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_gray_fib2_2024-01-30_11-10-53.pkl", "rb"))
    
    web_lk_bin1_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_binary_web1_2024-01-30_11-22-31.pkl", "rb"))
    web_lk_bin2_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_binary_web2_2024-01-30_11-25-56.pkl", "rb"))
    fib_lk_bin1_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_binary_fib1_2024-01-30_11-17-34.pkl", "rb"))
    fib_lk_bin2_raw = pickle.load(open("OF_outputs/data3_jan2023/LK_binary_fib2_2024-01-30_11-18-24.pkl", "rb"))
    
    # t_z_gnd load
    fib_z_1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/franka_raw/franka1_modified.csv',delimiter=',',header=None)
    fib_z_1 = fib_z_1.iloc[1:901,2].astype(float)
    # print('fib_z_1 shape: ',np.shape(fib_z_1))
    fib_z_2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/franka_raw/franka2_modified.csv',delimiter=',',header=None)
    fib_z_2 = fib_z_2.iloc[1:901,2].astype(float)
    web_z_1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/franka_raw/franka3_modified.csv',delimiter=',',header=None)
    web_z_1 = web_z_1.iloc[1:901,2].astype(float)
    web_z_2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/franka_raw/franka4_modified.csv',delimiter=',',header=None)
    web_z_2 = web_z_2.iloc[1:901,2].astype(float)
    
    # outputs for linking IO plots: 
    # webcam: 
    plt.figure(1)
    linkingIO(web_df_euler1,web_bd_gray1_raw,web_z_1) # for BD_gray
    plt.suptitle('Sample 1: Grayscale Webcam + Blob Detection')
    plt.figure(2)
    linkingIO(web_df_euler2,web_bd_gray2_raw,web_z_2)
    plt.suptitle('Sample 2: Grayscale Webcam + Blob Detection')
    
    plt.figure(3)
    linkingIO(web_df_euler1,web_bd_bin1_raw,web_z_1) # for BD_binary
    plt.suptitle('Sample 1: Binary Webcam + Blob Detection')
    plt.figure(4)
    linkingIO(web_df_euler2,web_bd_bin2_raw,web_z_2)
    plt.suptitle('Sample 2: Binary Webcam + Blob Detection')
    
    plt.figure(5)
    linkingIO(web_df_euler1,web_lk_gray1_raw,web_z_1) # for LK_gray
    plt.suptitle('Sample 1: Grayscale Webcam + Optical Flow')
    plt.figure(6)
    linkingIO(web_df_euler2,web_lk_gray2_raw,web_z_2)
    plt.suptitle('Sample 2: Grayscale Webcam + Optical Flow')
    
    plt.figure(7)
    linkingIO(web_df_euler1,web_lk_bin1_raw,web_z_1) # for LK_binary
    plt.suptitle('Sample 1: Binary Webcam + Optical Flow')
    plt.figure(8)
    linkingIO(web_df_euler2,web_lk_bin2_raw,web_z_2)
    plt.suptitle('Sample 2: Binary Webcam + Optical Flow')
    
    # fibrescope:
    plt.figure(9)
    linkingIO(fib_df_euler1,fib_bd_gray1_raw,fib_z_1) # for BD_gray
    plt.suptitle('Sample 1: Grayscale Fibrescope + Blob Detection')
    plt.figure(10)
    linkingIO(fib_df_euler2,fib_bd_gray2_raw,fib_z_2)
    plt.suptitle('Sample 2: Grayscale Fibrescope + Blob Detection')
    
    plt.figure(11)
    linkingIO(fib_df_euler1,fib_bd_bin1_raw,fib_z_1) # for BD_binary
    plt.suptitle('Sample 1: Binary Fibrescope + Blob Detection')
    plt.figure(12)
    linkingIO(fib_df_euler2,fib_bd_bin2_raw,fib_z_2)
    plt.suptitle('Sample 2: Binary Fibrescope + Blob Detection')
    
    plt.figure(13)
    linkingIO(fib_df_euler1,fib_lk_gray1_raw,fib_z_1) # for LK_gray
    plt.suptitle('Sample 1: Grayscale Fibrescope + Optical Flow')
    plt.figure(14)
    linkingIO(fib_df_euler2,fib_lk_gray2_raw,fib_z_2)
    plt.suptitle('Sample 2: Grayscale Fibrescope + Optical Flow')
    
    plt.figure(15)
    linkingIO(fib_df_euler1,fib_lk_bin1_raw,fib_z_1) # for LK_binary
    plt.suptitle('Sample 1: Binary Fibrescope + Optical Flow')
    plt.figure(16)
    linkingIO(fib_df_euler2,fib_lk_bin2_raw,fib_z_2)
    plt.suptitle('Sample 2: Binary Fibrescope + Optical Flow')
    
    plt.show()
    
    # save statistics to a csv file
    statistics_df = pd.DataFrame(statistics,columns=['Pearson Corr (lin)', 'Spearman Corr (non-lin)', 'R^2', 'Kendall Tau', 'kendall p-val', 'Weighted Tau'],dtype=float)
    statistics_df.to_csv('OF_outputs/statistics.csv')
    
if __name__ == "__main__":
    main()
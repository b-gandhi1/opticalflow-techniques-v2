import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt

# AIM: Link the OF and BD outputs to the ground truth in Euler format, and find a relationship between the two. 


# load pickle data from outputs folder 
def bd_data_loader_gray():
    bd_web = pickle.load( open( "OF_outputs/BD_gray_web1_2023-12-06_16-57-23.pkl", "rb" ) )
    max_len_web = max(len(d['data']) for d in bd_web)
    padded_web = [np.pad(d['data'], (0, max_len_web - len(d['data'])), mode='constant') for d in bd_web]
    data_bd_web = np.asarray(padded_web)
    
    bd_fibre = pickle.load( open( "OF_outputs/BD_gray_fib1_2023-12-07_12-30-19.pkl", "rb" ) )        
    max_len_fibre = max(len(d['data']) for d in bd_fibre)
    padded_fibre = [np.pad(d['data'], (0, max_len_fibre - len(d['data'])), mode='constant') for d in bd_fibre]
    data_bd_fibre = np.asarray(padded_fibre)
    
    # for i in range(len(data_bd_web)): # test shapes to identify inhomogenity
    #     print(data_bd_web[i].shape)
    
    return data_bd_web, data_bd_fibre

def bd_data_loader_binary():
    bd_web = pickle.load( open( "OF_outputs/BD_binary_web1_2023-12-06_16-49-54.pkl", "rb" ) )
    max_len_web = max(len(d['data']) for d in bd_web)
    padded_web = [np.pad(d['data'], (0, max_len_web - len(d['data'])), mode='constant') for d in bd_web]
    data_bd_web = np.asarray(padded_web)
    
    bd_fibre = pickle.load( open( "OF_outputs/BD_binary_fib1_2023-12-06_17-47-48.pkl", "rb" ) )
    max_len_fibre = max(len(d['data']) for d in bd_fibre)
    padded_fibre = [np.pad(d['data'], (0, max_len_fibre - len(d['data'])), mode='constant') for d in bd_fibre]
    data_bd_fibre = np.asarray(padded_fibre)
    
    return data_bd_web, data_bd_fibre

def lk_data_loader_gray():
    lk_web = pickle.load( open( "OF_outputs/LK_gray_web1_2023-12-04_17-02-48.pkl", "rb" ) )
    max_len_web = max(len(d['data']) for d in lk_web)
    padded_web = [np.pad(d['data'], (0, max_len_web - len(d['data'])), mode='constant') for d in lk_web]
    data_lk_web = np.asarray(padded_web)
    
    lk_fibre = pickle.load( open( "OF_outputs/LK_bright_fib1_2023-11-29_16-44-01.pkl", "rb" ) )
    max_len_fibre = max(len(d['data']) for d in lk_fibre)
    padded_fibre = [np.pad(d['data'], (0, max_len_fibre - len(d['data'])), mode='constant') for d in lk_fibre]
    data_lk_fibre = np.asarray(padded_fibre)
    
    return data_lk_web, data_lk_fibre

def lk_data_loader_binary():
    lk_web = pickle.load( open( "OF_outputs/LK_binary_web1_2023-12-06_17-56-36.pkl", "rb" ) )
    max_len_web = max(len(d['data']) for d in lk_web)
    padded_web = [np.pad(d['data'], (0, max_len_web - len(d['data'])), mode='constant') for d in lk_web]
    data_lk_web = np.asarray(padded_web)
    
    lk_fibre = pickle.load( open( "OF_outputs/LK_binary_fib1_2023-11-29_16-37-46.pkl", "rb" ) )
    max_len_fibre = max(len(d['data']) for d in lk_fibre)
    padded_fibre = [np.pad(d['data'], (0, max_len_fibre - len(d['data'])), mode='constant') for d in lk_fibre]
    data_lk_fibre = np.asarray(padded_fibre)
    
    return data_lk_web, data_lk_fibre


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
    
def linkingIO(gnd_truth_euler, fib_dat, web_dat):
    # get max displacements from fib_dat and web_dat
    
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
    
    # outputs for linking IO plots: 
    
    linkingIO() # for BD_gray
    linkingIO() # for BD_binary
    linkingIO() # for LK_gray
    linkingIO() # for LK_binary

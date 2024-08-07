import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import sys
from linkingIO import normalize_vector

# execution command example: python mapExpGnd.py pitch1 , python mapExpGnd.py roll1

exp_data_path = sys.argv[1]
length = len(exp_data_path)
pitchroll, i = exp_data_path[:length-1], exp_data_path[length-1] # split number at the endfrom filename 

if pitchroll == "pitch":
    path = "pitch_4-jun-2024/fibrescope"+i
    ax_sel, gnd_sel, imu_sel = 'y_vals', 'pitch_y', 'IMU X'
elif pitchroll == "roll":
    path = "roll_6-jun-2024/fibrescope"+i
    ax_sel, gnd_sel,imu_sel = 'x_vals', 'roll_x', 'IMU Y'
else:
    print("ERROR: Unrecognised input for pressure selector.")

dat_exp = pd.read_csv("imu-fusion-outputs/LK_"+pitchroll+"/imu-fusion-outputs_LK_Zavg"+exp_data_path+".csv",delimiter=',',usecols=[ax_sel],dtype={ax_sel: float}) # lk data, experimental data
dat_pressure = pd.read_csv("data_collection_with_franka/B07LabTrials/imu-sensor-fusion/"+path+".csv", delimiter=',',usecols=['Pressure (kPa)'],dtype={'Pressure (kPa)': float}) # feedback

dat_exp_norm = normalize_vector(dat_exp) # x normalized
dat_pressure_norm = normalize_vector(dat_pressure) # feedback normalized

dat_exp_pres_norm = pd.concat([dat_exp_norm,dat_pressure_norm],axis=1) # x with feedback (pressure)

dat_gnd_euler = pd.read_csv("imu-fusion-outputs/LK_"+pitchroll+"/"+exp_data_path+"euler_gnd.csv",delimiter=',',usecols=["pitch_y"],dtype={gnd_sel: float}) * (-1)# pitch_y being used for pitch and roll data both, works well. 
dat_gnd_euler_norm = normalize_vector(dat_gnd_euler) # gnd truth normalized
    
dat_gyro = pd.read_csv('data_collection_with_franka/B07LabTrials/imu-sensor-fusion/'+path+'.csv',delimiter=',',usecols=[imu_sel],dtype={imu_sel: float}) # feedback, *(-1) to fix orientations
dat_gyro_norm = normalize_vector(dat_gyro) # feedback normalized

# set offsets
if pitchroll == "pitch":
    offset_gnd_euler = dat_gnd_euler
    offset_gyro = dat_gyro
elif pitchroll == "roll": 
    offset_gnd_euler = -48.7 + dat_gnd_euler
    offset_gyro = -2.00 + dat_gyro
else: 
    print("ERROR: Unrecognised input for motion type selector.")
    exit()
    
# min max scaler for exp data 
min_val = offset_gnd_euler.min().iloc[0]
print("min_val scaler: ",min_val)
max_val = offset_gnd_euler.max().iloc[0]
print("max_val scaler: ",max_val)
scaler = MinMaxScaler(feature_range=(min_val,max_val))
# NOTE: partial_fit(X, y=None) -> for continuous stream of x, so RT live demos. 

scaled_dat_exp = scaler.fit_transform(dat_exp)
    
# plot the data
time = np.linspace(0,60,len(dat_exp_pres_norm))
plt.figure()
# plt.plot(time,dat_exp.iloc[:,0],label='experimental')
plt.plot(time,scaled_dat_exp,label='experimental')
plt.plot(time,offset_gnd_euler,label='gnd truth')
plt.plot(time,offset_gyro,label='gyro')
plt.legend()
plt.tight_layout()
plt.show()

# extract filtered data and save as csv files
save_bool = sys.argv[2]
if save_bool == "save":
    data = pd.concat([offset_gnd_euler,dat_pressure,offset_gyro, dat_exp],axis=1, ignore_index=False)
    savename = "imu-fusion-outputs/"+ pitchroll+"_concat/"+pitchroll+"_concat"+i+".csv"
    data.to_csv(savename, header=["GND (euler deg)","Pressure (kPa)","Gyro (deg)","Experimental (LK raw)"])
else: 
    print("Not saving...")
    pass

# explore relationship between data points, x and y


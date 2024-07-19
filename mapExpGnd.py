import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    ax_sel, gnd_sel, imu_sel = 'y_vals', 'pitch_y', 'IMU Y'
elif pitchroll == "roll":
    path = "roll_6-jun-2024/fibrescope"+i
    ax_sel, gnd_sel,imu_sel = 'x_vals', 'roll_x', 'IMU X'
else:
    print("ERROR: Unrecognised input for pressure selector.")

dat_exp = pd.read_csv("imu-fusion-outputs/LK_"+pitchroll+"/imu-fusion-outputs_LK_Zavg"+exp_data_path+".csv",delimiter=',',usecols=[ax_sel],dtype={ax_sel: float}) # x
dat_pressure = pd.read_csv("data_collection_with_franka/B07LabTrials/imu-sensor-fusion/"+path+".csv", delimiter=',',usecols=['Pressure (kPa)'],dtype={'Pressure (kPa)': float}) # feedback

dat_exp_norm = normalize_vector(dat_exp) # x normalized
dat_pressure_norm = normalize_vector(dat_pressure) # feedback normalized

dat_exp_pres_norm = pd.concat([dat_exp_norm,dat_pressure_norm],axis=1) # x with feedback (pressure)

dat_gnd_euler = pd.read_csv("imu-fusion-outputs/LK_"+pitchroll+"/"+exp_data_path+"euler_gnd.csv",delimiter=',',usecols=[gnd_sel],dtype={gnd_sel: float}) # y
dat_gnd_euler_norm = normalize_vector(dat_gnd_euler) # y normalized

dat_gyro = pd.read_csv('data_collection_with_franka/B07LabTrials/imu-sensor-fusion/'+path+'.csv',delimiter=',',usecols=[imu_sel],dtype={imu_sel: float}) # feedback
dat_gyro_norm = normalize_vector(dat_gyro) # feedback normalized
# plot the data
time = np.linspace(0,60,len(dat_exp_pres_norm))
plt.figure()
# plt.plot(time,dat_exp_pres_norm.iloc[:,0])
plt.plot(time,dat_gnd_euler)
plt.plot(time,dat_gyro)
plt.legend(['experimental','gnd truth','gyro'])
plt.tight_layout()
plt.show()

# explore relationship between data points, x and y


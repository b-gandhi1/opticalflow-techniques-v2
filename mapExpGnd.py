import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys
from linkingIO import normalize_vector

exp_data_path = sys.argv[1]
length = len(exp_data_path)
pressure_sel, i = exp_data_path[:length-1], exp_data_path[length-1] # split number at the endfrom filename 

if pressure_sel == "pitch":
    pressure_path = "pitch_4-jun-2024/fibrescope"+i
elif pressure_sel == "roll":
    pressure_path = "roll_6-jun-2024/fibrescope"+i
else:
    print("ERROR: Unrecognised input for pressure selector.")

dat_exp = pd.read_csv("imu-fusion-outputs/LK_pitch/imu-fusion-outputs_LK_Zavg"+exp_data_path+".csv",delimiter=',',usecols=['x_vals','y_vals','z_vals'],dtype={'x_vals': float,'y_vals': float,'z_vals': float}) # x
dat_pressure = pd.read_csv("data_collection_with_franka/B07LabTrials/imu-sensor-fusion/"+pressure_path+".csv", delimiter=',',usecols=['Pressure (kPa)'],dtype={'Pressure (kPa)': float}) # feedback

dat_exp_norm = normalize_vector(dat_exp) # x normalized
dat_pressure_norm = normalize_vector(dat_pressure) # feedback normalized

dat_exp_pres_norm = pd.concat([dat_exp_norm,dat_pressure_norm],axis=1) # x with feedback (pressure)

dat_gnd_euler = pd.read_csv("imu-fusion-outputs/LK_pitch/"+exp_data_path+"euler_gnd.csv",delimiter=',',usecols=['roll_x','pitch_y','yaw_z'],dtype={'x': float,'y': float,'z': float}) # y
dat_gnd_euler_norm = normalize_vector(dat_gnd_euler) # y normalized

# plot the data
time = np.linspace(0,60,len(dat_exp_pres_norm))
plt.figure()
plt.plot(time,dat_exp_pres_norm)
plt.plot(time,dat_gnd_euler_norm)
plt.legend(['experimental','gnd truth'])
plt.tight_layout()
plt.show()

# explore relationship between data points, x and y



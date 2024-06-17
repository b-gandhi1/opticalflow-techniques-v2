import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys
from linkingIO import normalize_vector

path = sys.argv[1]

dat_exp = pd.read_csv() # x
dat_pressure = pd.read_csv() # feedback

dat_exp_norm = normalize_vector(dat_exp) # x normalized
dat_pressure_norm = normalize_vector(dat_pressure) # feedback normalized

dat_exp_pres_norm = pd.concat([dat_exp_norm,dat_pressure_norm],axis=1) # x with feedback (pressure)

dat_gnd_euler = pd.read_csv() # y
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



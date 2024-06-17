import pickle
import pandas as pd
import numpy as np
import sys
import os

path_pkl = sys.argv[1]
savefile = sys.argv[2]
# load data from cams

pickle_load = pickle.load(open(path_pkl,"rb"))

# extract 1D xyz vars

x_vals = np.asarray([data['x_val_1d'] for data in pickle_load],dtype=float)
y_vals = np.asarray([data['y_val_1d'] for data in pickle_load],dtype=float)
z_vals = np.asarray([data['z_val'] for data in pickle_load],dtype=float) # 1d z vars 

# put it all on a csv file and save it

path_save = os.path.dirname(path_pkl)
pd = pd.DataFrame({'x_vals':x_vals,'y_vals':y_vals,'z_vals':z_vals},dtype=float)
pd.to_csv(os.path.join(path_save,'imu-fusion-outputs_LK_Zavg'+savefile+'.csv'),header=True)

print("--- END ---")

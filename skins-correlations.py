from scipy.stats import spearmanr
import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob 

# inputs: 
pitchrollN = sys.argv[1]
pitchroll, num = pitchrollN[0:len(pitchrollN)-1], pitchrollN[-1]
spacing = sys.argv[2]

# define paths
exp_path = "skins-test-outputs/cm"+spacing+"/"+pitchroll+"/skins-"+pitchrollN+".csv"
gnd_path = glob.glob("data_collection_with_franka/B07LabTrials/skins-data/cm"+spacing+"/"+pitchroll+"/fibrescope*"+num+"*gnd.csv")[0]

# load data
experimental_data_all = pd.read_csv(exp_path, delimiter=',', dtype={'x_vals': float, 'y_vals': float, 'z_vals': float}, skiprows=[i for i in range(1,13)])
gnd_data_all = pd.read_csv(gnd_path, delimiter=',', dtype={'roll_x': float, 'pitch_y': float, 'yaw_z': float})

if pitchroll == "pitch":
    exp_ax = 'x_vals'
    gnd_ax = 'roll_x'
elif pitchroll == "roll":
    exp_ax = 'y_vals'
    gnd_ax = 'pitch_y'
else:
    print("Invalid pitchroll input")
    sys.exit(1)

experimental_data = experimental_data_all.loc[exp_ax]
ground_truth = gnd_data_all.loc[gnd_ax]

# calculate correlations
corr_nonlin, _ = spearmanr(experimental_data, ground_truth, alternative='two-sided', nan_policy='propagate') # spearman correlation (nonlinear). High corr ~= 1. -1 < corr_range < 1. same as pearson.

# test plots
plt.figure()
plt.plot(experimental_data, label='Exp_dat')
plt.plot(ground_truth, label='Gnd_dat')
plt.legend()
plt.title(pitchrollN+" corr: "+str(corr_nonlin))

print("Spearman correlation: ", corr_nonlin)

# tabulate correlations

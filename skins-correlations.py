from scipy.stats import spearmanr
import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob 
from sklearn.preprocessing import MinMaxScaler, normalize

def corr_calc(pitchrollN, pitchroll, num, spacing): 

    # define paths
    exp_path = "skins-test-outputs/cm"+spacing+"/"+pitchroll+"/skins-"+pitchrollN+".csv"
    gnd_path = glob.glob("data_collection_with_franka/B07LabTrials/skins-data/cm"+spacing+"/"+pitchroll+"/fibrescope*"+num+"*gnd.csv")[0]

    # load data
    experimental_data_all = pd.read_csv(exp_path, delimiter=',', dtype={'x_vals': float, 'y_vals': float, 'z_vals': float}, skiprows=[i for i in range(1,13)])
    # print(experimental_data_all.columns)
    gnd_data_all = pd.read_csv(gnd_path, delimiter=',', dtype={'roll_x': float, 'pitch_y': float, 'yaw_z': float})
    # print(gnd_data_all.columns)

    if pitchroll == "pitch":
        exp_ax = 'x_vals'
        gnd_ax = 'roll_x'
        offset_gnd = 58.0
    elif pitchroll == "roll":
        exp_ax = 'y_vals'
        gnd_ax = 'pitch_y'
        offset_gnd = -46.0 
    else:
        print("Invalid pitchroll input")
        sys.exit(1)

    experimental_data = experimental_data_all[exp_ax]
    print("max = ", experimental_data.max(), "    min = ", experimental_data.min())
    ground_truth = gnd_data_all[gnd_ax] 

    offset_gnd_dat = ground_truth + offset_gnd

    norm_exp_data = (experimental_data.values - experimental_data.min())/(experimental_data.max() - experimental_data.min()) * (offset_gnd_dat.max() - offset_gnd_dat.min()) + offset_gnd_dat.min()
    # scaler = MinMaxScaler()
    # scaler.fit(offset_gnd_dat.values.reshape(-1,1))
    # scaled_exp_data = scaler.transform(experimental_data.values.reshape(-1,1))
    # scaled_exp_data = pd.Series(scaled_exp_data.flatten())

    # calculate correlations
    corr_nonlin, _ = spearmanr(experimental_data, ground_truth, alternative='two-sided', nan_policy='propagate') # spearman correlation (nonlinear). High corr ~= 1. -1 < corr_range < 1. same as pearson.

    # test plots
    plt.figure()
    plt.plot(norm_exp_data, label='Exp_dat')
    plt.plot(offset_gnd_dat, label='Gnd_dat')
    plt.legend()
    plt.title(pitchrollN+" corr: "+str(corr_nonlin))
    plt.show() 

    print("Spearman correlation: ", corr_nonlin)
    
    return corr_nonlin

def main(pitchrollN, pitchroll, num, spacing): 
    # for loop to go through all datasets
    corr = corr_calc(pitchrollN, pitchroll, num, spacing)
    # tabulate correlations in for loop 
    
if __name__ == "__main__":
    
    # inputs: 
    pitchrollN = sys.argv[1]
    pitchroll, num = pitchrollN[0:len(pitchrollN)-1], pitchrollN[-1]
    spacing = sys.argv[2]
    
    main(pitchrollN, pitchroll, num, spacing)

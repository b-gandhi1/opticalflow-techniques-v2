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
    plt.title(pitchrollN+" "+spacing+" corr: "+str(corr_nonlin))
    # plt.show() 

    print("Spearman correlation ("+pitchrollN+" "+spacing+"): ", corr_nonlin)
    
    return corr_nonlin

def main(): 
    
    corr_df = pd.DataFrame(columns=['pitchrollN', 'Spacing', 'Correlation'], index=None)
    
    # for loop to go through all datasets
    for spacing in ["0-5", "1-0", "1-5"]:
        for pitchroll in ["pitch", "roll"]: 
            for num in range(1,5+1):
                num = str(num)
                pitchrollN = pitchroll+num
                corr = corr_calc(pitchrollN, pitchroll, num, spacing)
                # print("Correlation for "+pitchrollN+" spacing "+spacing+" is: "+str(corr))
                new_row = pd.DataFrame({'pitchrollN': pitchrollN, 'Spacing': spacing, 'Correlation': corr}, index=[0])
                corr_df = pd.concat([corr_df, new_row], ignore_index=True)
    
    print(corr_df)
    corr_df.to_csv('skins-test-outputs/skins-correlations.csv', index=False)
    plt.show()
        
if __name__ == "__main__":
    
    # inputs: 
    # pitchrollN = sys.argv[1]
    # pitchroll, num = pitchrollN[0:len(pitchrollN)-1], pitchrollN[-1]
    # spacing = sys.argv[2]
    try: 
        main()
    except KeyboardInterrupt:
        plt.close()
        sys.exit(1)

from scipy.stats import spearmanr
import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob 
import numpy as np

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
        offset_gnd = 46.0 
        gnd_data_all[gnd_ax] = gnd_data_all[gnd_ax] * (-1)
        
    else:
        print("Invalid pitchroll input")
        sys.exit(1)

    experimental_data = experimental_data_all[exp_ax]
    # print("max = ", experimental_data.max(), "    min = ", experimental_data.min())
    ground_truth = gnd_data_all[gnd_ax] 

    offset_gnd_dat = ground_truth + offset_gnd

    norm_exp_data = (experimental_data.values - experimental_data.min())/(experimental_data.max() - experimental_data.min()) * (offset_gnd_dat.max() - offset_gnd_dat.min()) + offset_gnd_dat.min()
    # scaler = MinMaxScaler()
    # scaler.fit(offset_gnd_dat.values.reshape(-1,1))
    # scaled_exp_data = scaler.transform(experimental_data.values.reshape(-1,1))
    # scaled_exp_data = pd.Series(scaled_exp_data.flatten())

    # calculate correlations
    corr_nonlin, _ = spearmanr(experimental_data, ground_truth, alternative='two-sided', nan_policy='propagate') # spearman correlation (nonlinear). High corr ~= 1. -1 < corr_range < 1. same as pearson.
    
    mean_abs_error = abs(offset_gnd_dat - norm_exp_data).mean() # MAE
    
    # test plots
    t = np.linspace(0, 60, len(norm_exp_data))
    plt.figure()
    plt.plot(t,norm_exp_data, label='Exp_dat')
    plt.plot(t,offset_gnd_dat, label='Gnd_dat')
    plt.ylabel('Angle (degrees)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title(pitchrollN+" "+spacing+" corr: "+str(corr_nonlin)+", mean abs error: "+str(mean_abs_error))
    # plt.show() 

    # print("Spearman correlation ("+pitchrollN+" "+spacing+"): ", corr_nonlin)
    
    
    return corr_nonlin, mean_abs_error

def cross_corr():

    df = pd.DataFrame(columns=['Pitch|Roll N','Corr w 5mm', 'Corr w 15mm'], index=None)
    for skinspitchroll in ["skins-pitch", "skins-roll"]:
        spacing10 = glob.glob("skins-test-outputs/cm1-0/**/"+skinspitchroll+"*.csv") # default skin
        print(spacing10)
        if skinspitchroll == "skins-pitch":
            ax = "x_vals"
        elif skinspitchroll == "skins-roll":
            ax = "y_vals"
        else:
            print("ERROR: Invalid pitchroll, exiting script.")
            sys.exit(1)
            
        for i in range(0,5):
            def_dat = pd.read_csv(spacing10[i], delimiter=',', usecols={ax}, dtype={ax: float}, skiprows=[j for j in range(1,13)])
            print("def dat: ", def_dat.size, ' ',i)
            for k in range(1,5+1):            
                compareN5 = glob.glob("skins-test-outputs/cm0-5/**/"+skinspitchroll+str(k)+".csv")[0]
                # print("compareN5: ",compareN5)
                comp5dat = pd.read_csv(compareN5, delimiter=',', usecols={ax}, dtype={ax: float}, skiprows=[j for j in range(1,13)])
                compareN15 = glob.glob("skins-test-outputs/cm1-5/**/"+skinspitchroll+str(k)+".csv")[0]
                # print("compareN15: ",compareN15)
                comp15dat = pd.read_csv(compareN15, delimiter=',', usecols={ax}, dtype={ax: float}, skiprows=[j for j in range(1,13)])
                
                corr_nonlin5, _ = spearmanr(comp5dat, def_dat, alternative='two-sided', nan_policy='propagate') 
                corr_nonlin15, _ = spearmanr(comp15dat, def_dat, alternative='two-sided', nan_policy='propagate')
                
                # print("corr5: ", corr_nonlin5, "corr15: ", corr_nonlin15)
                newrow = pd.DataFrame({'Pitch|Roll N': skinspitchroll+str(k), 'Corr w 5mm': corr_nonlin5, 'Corr w 15mm': corr_nonlin15}, index=[0])
                df = pd.concat([df, newrow], ignore_index=True)
            
    return df
    
def main(): 
    
    df = pd.DataFrame(columns=['pitchrollN', 'Spacing', 'Correlation', 'MAE'], index=None)
    
    # for loop to go through all datasets
    for spacing in ["0-5", "1-0", "1-5"]:
        for pitchroll in ["pitch", "roll"]: 
            for num in range(1,5+1):
                num = str(num)
                pitchrollN = pitchroll+num
                corr, mean_abs_err = corr_calc(pitchrollN, pitchroll, num, spacing)
                # print("Correlation for "+pitchrollN+" spacing "+spacing+" is: "+str(corr))
                new_row = pd.DataFrame({'pitchrollN': pitchrollN, 'Spacing': spacing, 'Correlation': corr, 'MAE': mean_abs_err}, index=[0])
                df = pd.concat([df, new_row], ignore_index=True)
    
    # df.to_csv('skins-test-outputs/skins-correlations.csv', index=False) # uncomment to save new csv    
    coss_corr_df = cross_corr() # spacings
    coss_corr_df.to_csv('skins-test-outputs/skins-cross-correlations.csv', index=False) # uncomment to save new csv
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

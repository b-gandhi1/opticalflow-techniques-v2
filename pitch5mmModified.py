from scipy.stats import spearmanr
import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob 
import numpy as np
# from skins_correlations import hysteresis_test

FPS = 10 # Hz

pitchroll, spacing = "pitch", "0-5"
threshold = 9.0 # +- 10 degrees range of motion

def hysteresis_test(pitchrollN, spacing, pitchroll, time_period, norm_exp_data, offset_gnd_dat):

    # parameters for pitch only. 
    time_offset = 55
    window_len = int(time_period/4 * FPS)
    norm_exp_data = abs(norm_exp_data[time_offset:])
    offset_gnd_dat = abs(offset_gnd_dat[time_offset:])
    # print("Hysteresis test for pitch window: ", window_len)

    loading_df = pd.DataFrame(index=None)
    gnd_load_df = pd.DataFrame(index=None)
    unloading_df = pd.DataFrame(index=None)
    gnd_unload_df = pd.DataFrame(index=None)
    new_load_row = pd.DataFrame(index=None)
    gnd_new_load_row = pd.DataFrame(index=None)
    new_unload_row = pd.DataFrame(index=None)
    gnd_new_unload_row = pd.DataFrame(index=None)
    # seperate norm_exp_data using moving window sizes for loading and unloading
    check=0
    for i in range(0, len(norm_exp_data), window_len):
        if check % 2 == 0: # loading data+
            # print('0 modulus')
            new_load_row = pd.DataFrame([norm_exp_data[i:i+window_len]], index=None)
            gnd_new_load_row = pd.DataFrame([offset_gnd_dat[i+1:i+1+window_len]], index=None)
        
        else: # unloading data
            # print('1 modulus')
            # breakpoint()
            new_unload_row = pd.DataFrame([norm_exp_data[i:i+window_len]], index=None)
            gnd_new_unload_row = pd.DataFrame([offset_gnd_dat[i:i+window_len]], index=None)
            
        loading_df = pd.concat([loading_df, new_load_row], ignore_index=True, axis=0)
        gnd_load_df = pd.concat([gnd_load_df, gnd_new_load_row], ignore_index=True, axis=0)
        unloading_df = pd.concat([unloading_df, new_unload_row], ignore_index=True, axis=0)
        gnd_unload_df = pd.concat([gnd_unload_df, gnd_new_unload_row], ignore_index=True, axis=0)
        check+=1
    t_load = np.linspace(0, loading_df.shape[1]/FPS, loading_df.shape[1])
    t_unload = np.linspace(0, unloading_df.shape[1]/FPS, unloading_df.shape[1])
    loading_avg = loading_df.mean(axis=0)
    load_neg_err = loading_df.mean(axis=0) - loading_df.min(axis=0)
    load_pos_err = loading_df.max(axis=0) - loading_df.mean(axis=0)
    gnd_load_avg = gnd_load_df.mean(axis=0)
    unloading_avg = unloading_df.mean(axis=0)
    unload_neg_err = unloading_df.mean(axis=0) - unloading_df.min(axis=0)
    unload_pos_err = unloading_df.max(axis=0) - unloading_df.mean(axis=0)
    gnd_unload_avg = gnd_unload_df.mean(axis=0)

    fig,(ax1,ax2) = plt.subplots(2,1)
    ax1.errorbar(t_load, loading_avg, fmt = 'b*', label='MCP')
    ax2.errorbar(t_unload, unloading_avg, fmt = 'bo', label='unloading')
    ax1.plot(t_load, gnd_load_avg, 'b--', label='Gnd')
    ax2.plot(t_unload, gnd_unload_avg, 'b--',  label='Gnd')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Loading Angle (deg)')
    ax1.legend()
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Unloading Angle (deg)')
    ax2.legend()
    

    fig.canvas.manager.set_window_title(pitchrollN+" "+spacing+" cm - hysteresis")
    # plt.show()
    
    errbar_dat = pd.DataFrame({'Time': t_load, 'Loading_avg': loading_avg, 'Loading_neg_err': load_neg_err, 'Loading_pos_err': load_pos_err, 'Gnd_load_avg': gnd_load_avg ,'Unloading_avg': unloading_avg, 'Unloading_neg_err': unload_neg_err, 'Unloading_pos_err': unload_pos_err, 'Gnd_unload_avg': gnd_unload_avg})
    # save loading and unloading data
    loading_df.to_csv('skins-test-outputs/hysteresis/cm'+spacing+'/'+pitchrollN+'-loading.csv', index=False)
    gnd_load_df.to_csv('skins-test-outputs/hysteresis/cm'+spacing+'/'+pitchrollN+'-gnd-loading.csv', index=False)
    unloading_df.to_csv('skins-test-outputs/hysteresis/cm'+spacing+'/'+pitchrollN+'-unloading.csv', index=False)
    gnd_unload_df.to_csv('skins-test-outputs/hysteresis/cm'+spacing+'/'+pitchrollN+'-gnd-unloading.csv', index=False)
    errbar_dat.to_csv('skins-test-outputs/hysteresis/cm'+spacing+'/'+pitchrollN+'-errbar.csv', index=False)

def main():
    df = pd.DataFrame(index=None)
    for num in range(1,5+1):
        num = str(num)
        pitchrollN = pitchroll+num
        exp_path = "skins-test-outputs/cm"+spacing+"/"+pitchroll+"/skins-"+pitchrollN+".csv"
        # breakpoint()
        gnd_path = glob.glob("data_collection_with_franka/B07LabTrials/skins-data/cm"+spacing+"/"+pitchroll+"/fibrescope*"+num+"*gnd.csv")[0]

        experimental_data_all = pd.read_csv(exp_path, delimiter=',', dtype={'x_vals': float}, usecols={'x_vals'}, skiprows=[i for i in range(1,13)]).values
        gnd_data_all = pd.read_csv(gnd_path, delimiter=',', dtype={'roll_x': float}, usecols={'roll_x'}).values
        offset_gnd = 37.9
        time_period = 12.3
        
        offset_gnd_dat = gnd_data_all + offset_gnd        
        norm_exp_data = (experimental_data_all - experimental_data_all.min())/(experimental_data_all.max() - experimental_data_all.min()) * (offset_gnd_dat.max() - offset_gnd_dat.min()) + offset_gnd_dat.min()
        norm_exp_mod_dat, gnd_mod_dat = [], []
        for i in range(0, len(norm_exp_data)):
            if abs(norm_exp_data[i]) < threshold:
                norm_exp_mod_dat.append(norm_exp_data[i])
                gnd_mod_dat.append(offset_gnd_dat[i])
        
        exp_df = np.array(norm_exp_mod_dat).flatten()
        gnd_df = np.array(gnd_mod_dat).flatten()
        corr_nonlin, _ = spearmanr(exp_df, gnd_df, alternative='two-sided', nan_policy='propagate') # spearman correlation (nonlinear). High corr ~= 1. -1 < corr_range < 1. same as pearson.
        mean_abs_error = abs(gnd_df - exp_df).mean() # MAE
        # motion_range = norm_exp_data.max() - norm_exp_data.min() # peak to peak range of motion
        # resolution = motion_range/(time_period*FPS) # range of motion / number of samples in a time period
        # breakpoint()
        hysteresis_test(pitchrollN+"mod", spacing, pitchroll, time_period, exp_df, gnd_df)
        
        # print("Correlation for "+pitchrollN+" spacing "+spacing+" is: "+str(corr))
        new_row = pd.DataFrame({'pitchrollN': pitchrollN, 'Spacing': spacing, 'Correlation': corr_nonlin, 'MAE': mean_abs_error}, index=[0])
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv('skins-test-outputs/pitch5mmModified-corr.csv', index=False)
    print(df)
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
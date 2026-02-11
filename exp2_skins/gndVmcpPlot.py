import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd

FPS = 10 # Hz

spacing = ["0-5", "1-0", "1-5"] # cm, spacings for respective path generations

# pitch vars 
exp_ax_pitch = 'x_vals'
gnd_ax_pitch = 'roll_x'
offset_gnd_pitch = 37.9

# roll vars 
exp_ax_roll = 'y_vals'
gnd_ax_roll = 'pitch_y'
offset_gnd_roll = 46.0 

trim = 12 # for mcp vals

# for each path, plot MCP val against ground truth onto the same plot, colour coded. 
for space in spacing: 
    path_mcp_list_pitch = sorted(glob.glob(f"skins-test-outputs/cm{space}/pitch/skins-pitch*.csv"))
    path_gnd_list_pitch = sorted(glob.glob(f"data_collection_with_franka/B07LabTrials/skins-data/cm{space}/pitch/fibrescope*euler_gnd.csv"))
    path_mcp_list_roll = sorted(glob.glob(f"skins-test-outputs/cm{space}/roll/skins-roll*.csv"))
    path_gnd_list_roll = sorted(glob.glob(f"data_collection_with_franka/B07LabTrials/skins-data/cm{space}/roll/fibrescope*euler_gnd.csv"))
    
    sampl_counter = 1
    plt.figure()
    # pitch graphs
    for path_mcp, path_gnd in zip(path_mcp_list_pitch, path_gnd_list_pitch):
        
        mcp_data_pitch = pd.read_csv(path_mcp, usecols={exp_ax_pitch})
        gnd_data_pitch = pd.read_csv(path_gnd, usecols={gnd_ax_pitch}) + offset_gnd_pitch
        mcp_data_pitch = mcp_data_pitch.iloc[trim:]
        
        pitch_norm = (mcp_data_pitch.values - mcp_data_pitch.values.min())/(mcp_data_pitch.values.max() - mcp_data_pitch.values.min()) * (gnd_data_pitch.values.max() - gnd_data_pitch.values.min()) + gnd_data_pitch.values.min()
        # breakpoint()
        plt.subplot(1, 2, 1)
        # plt.plot(pitch_norm, label='MCP')
        # plt.plot(gnd_data_pitch, label='Ground Truth')
        # plt.title(f"Pitch - {space} cm")
        # plt.xlabel('Time (s)')
        plt.scatter(pitch_norm, gnd_data_pitch.values, label=f"{sampl_counter}")
        plt.xlabel('MCP, pitch (deg)')
        plt.ylabel('Ground Truth, pitch (deg)')
        sampl_counter += 1
        plt.legend()
        plt.tight_layout()
        
    sampl_counter = 1 # reset counter
    # roll graphs
    for path_mcp, path_gnd in zip(path_mcp_list_roll, path_gnd_list_roll):
        # roll data
        mcp_data_roll = pd.read_csv(path_mcp, usecols={exp_ax_roll})
        gnd_data_roll = (pd.read_csv(path_gnd, usecols={gnd_ax_roll}) *(-1)) + offset_gnd_roll
        mcp_data_roll = mcp_data_roll.iloc[trim:]
        
        roll_norm = (mcp_data_roll.values - mcp_data_roll.values.min())/(mcp_data_roll.values.max() - mcp_data_roll.values.min()) * (gnd_data_roll.values.max() - gnd_data_roll.values.min()) + gnd_data_roll.values.min()
        plt.subplot(1, 2, 2)
        # plt.plot(roll_norm, label='MCP')
        # plt.plot(gnd_data_roll, label='Ground Truth')
        # plt.title(f"Roll - {space} cm")
        # plt.xlabel('Time (s)')
        plt.scatter(roll_norm, gnd_data_roll.values, label=f"{sampl_counter}")
        plt.xlabel('MCP, roll (deg)')
        plt.ylabel('Ground Truth, roll (deg)')
        sampl_counter += 1
        plt.legend()
        plt.tight_layout()
        
plt.show()



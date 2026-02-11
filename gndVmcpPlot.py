import matplotlib.pyplot as plt
import glob
import sys
import os
import numpy as np
import pandas as pd

FPS = 10 # Hz

spacing = ["0-5", "1-0", "1-5"] # cm, spacings for respective path generations

# for each path, plot MCP val against ground truth onto the same plot, colour coded. 
for space in spacing: 
    path_mcp_list = glob.glob(f"skins-test-outputs/cm{space}/pitch/skins-pitch*.csv", sorted=True)
    path_gnd_list = glob.glob(f"data_collection_with_franka/B07LabTrials/skins-data/cm{space}/pitch/fibrescope*.csv", sorted=True)

    plt.figure()
    for path_mcp, path_gnd in zip(path_mcp_list, path_gnd_list):
        mcp_data = pd.read_csv(path_mcp)
        gnd_data = pd.read_csv(path_gnd)

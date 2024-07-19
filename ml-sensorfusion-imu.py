import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
# from sklearn import svm
from sklearn.linear_model import LinearRegression # for a trial
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, roc_curve, RocCurveDisplay, auc)
from sklearn.multioutput import MultiOutputRegressor
# use torch instead of sklearn, numpy, and pandas - more efficient. 
import torch 
import sys

# use for loops in data loader functions to gather data from all files! 

def imu_data_loader(imu_sel,path): # load imu data and pressure vals from csv
    gyro_data = pd.read_csv('data_collection_with_franka/B07LabTrials/imu-sensor-fusion/'+path+'.csv',delimiter=',',usecols=[imu_sel],dtype={imu_sel: float})
    pressure_data = pd.read_csv("data_collection_with_franka/B07LabTrials/imu-sensor-fusion/"+path+".csv", delimiter=',',usecols=['Pressure (kPa)'],dtype={'Pressure (kPa)': float})
    
    
def lk_data_loader(ax_sel,path):
    lk_data = pd.read_csv("imu-fusion-outputs/LK_pitch/imu-fusion-outputs_LK_Zavg"+path+".csv",delimiter=',',usecols=[ax_sel],dtype={ax_sel: float})
    
def franka_euler_loader(gnd_sel,pitchroll,path):
    gnd_franka_data = pd.read_csv("imu-fusion-outputs/LK_"+pitchroll+"/"+path+"euler_gnd.csv",delimiter=',',usecols=[gnd_sel],dtype={gnd_sel: float}) 
    
def plots():
    ...
    
def ml_model(pitchroll, mcpX_train, mcp_y_train, mcpX_test, mcp_y_test, mcp_imuX_train, mcp_imu_y_train, mcp_imuX_test, mcp_imu_y_test):
    
    gnb = GaussianNB()
    
    mcp_gnb = MultiOutputRegressor(gnb) # define model LK map
    mcp_imu_gnb = MultiOutputRegressor(gnb) # define model LK+imu map
    
    pipe_mcp = make_pipeline(StandardScaler(), mcp_gnb)
    pipe_mcp_imu = make_pipeline(StandardScaler(), mcp_imu_gnb)
    
    pipe_mcp.fit(mcpX_train, mcp_y_train)
    pipe_mcp_imu.fit(mcp_imuX_train, mcp_imu_y_train)
    
    y_pred_mcp = pipe_mcp.predict(mcpX_test)
    y_pred_mcp_imu = pipe_mcp_imu.predict(mcp_imuX_test)
    
    f1_mcp = f1_score(mcp_y_test, y_pred_mcp, average='weighted')
    f1_mcp_imu = f1_score(mcp_imu_y_test, y_pred_mcp_imu, average='weighted')
    print("Metric:"+'\t'+"MCP" + '\t' + "MCP+IMU")
    print("Accuracy: ", str(accuracy_score(mcp_y_test, y_pred_mcp)) + '\t' + str(accuracy_score(mcp_imu_y_test, y_pred_mcp_imu)))
    print("F1 score: ", str(f1_mcp) + '\t' + str(f1_mcp_imu))
    
    # confusion matrices
    
    # ROC curve
    
    # cross validations
    
    # save trained models
    
    ...

def main():
    ...
    
if __name__ == "__main__":
    exp_data_path = sys.argv[1]
    length = len(exp_data_path)
    pitchroll, i = exp_data_path[:length-1], exp_data_path[length-1] # split number at the endfrom filename 

    if pitchroll == "pitch":
        path = "pitch_4-jun-2024/fibrescope"+i
        ax_sel, gnd_sel, imu_sel = 'y_vals', 'pitch_y', 'IMU Y'
    elif pitchroll == "roll":
        path = "roll_6-jun-2024/fibrescope"+i
        ax_sel, gnd_sel,imu_sel = 'x_vals', 'roll_x', 'IMU X'
    else:
        print("ERROR: Unrecognised input for pressure selector.")
    
    imu_data, pressure_data = imu_data_loader(imu_sel,path)
    lk_data = lk_data_loader(ax_sel,path)
    gnd_franka_data = franka_euler_loader(gnd_sel,pitchroll,exp_data_path)
    
    main(imu_data, pressure_data, lk_data, gnd_franka_data)
    
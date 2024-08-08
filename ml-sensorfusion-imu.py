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
# import torch 
import sys
import glob

# use for loops in data loader functions to gather data from all files! 

    
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
    
    pitchrolli = sys.argv[1]
    length = len(pitchrolli)
    pitchroll, i = pitchrolli[:length-1], pitchrolli[length-1] # split number at the endfrom filename 

    if pitchroll == "pitch":
        # path = "imu-fusion-outputs/pitch_concat/pitch_concat"+i+".csv"
        path = "imu-fusion-outputs/pitch_concat/pitch_concat"
    elif pitchroll == "roll":
        # path = "imu-fusion-outputs/roll_concat/roll_concat"+i+".csv"
        path = "imu-fusion-outputs/roll_concat/roll_concat"
    else:
        print("ERROR: Unrecognised input for pressure selector.")
    
    csvfiles = glob.glob(path+"*.csv")
    data = pd.read_csv(path, delimiter=',',dtype={'GND (euler deg)': float,'Pressure (kPa)': float,'Gyro (deg)': float,'Experimental (LK raw)': float})
    
    data_combined = pd.concat([pd.read_csv(f) for f in csvfiles])
    data_train, data_test = train_test_split(data_combined, test_size=0.2)
    
    main(data_train, data_test)
    
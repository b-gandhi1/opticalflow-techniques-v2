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
import glob

# use for loops in data loader functions to gather data from all files! 

def plots():
    ...
    
def ml_model(useimu, mcpX_train, mcp_y_train, mcpX_test, mcp_y_test):
    
    gnb = GaussianNB()
    
    mcp_gnb = MultiOutputRegressor(gnb) # define model LK map
    
    pipe_mcp = make_pipeline(StandardScaler(), mcp_gnb)
    
    pipe_mcp.fit(mcpX_train, mcp_y_train)
    
    y_pred_mcp = pipe_mcp.predict(mcpX_test)
    
    f1_mcp = f1_score(mcp_y_test, y_pred_mcp, average='weighted')
    print("Metric:"+'\t'+"MCP" + '\t' + "MCP+IMU")
    print("Accuracy: ", str(accuracy_score(mcp_y_test, y_pred_mcp)))
    print("F1 score: ", str(f1_mcp))
    
    # confusion matrices
    
    # ROC curve
    
    # cross validations
    
    # save trained models
    if useimu == "no-imu":
        torch.save(pipe_mcp, 'ml-models/mcp_no-imu_model.pth')
    elif useimu == "use-imu":
        torch.save(pipe_mcp, 'ml-models/mcp_imu_model.pth')
    else:
        print("ERROR: Unrecognised input.")

def main(useimu):
    
    # if pitchroll == "pitch":
    #     # path = "imu-fusion-outputs/pitch_concat/pitch_concat"+i+".csv"
    #     path = "imu-fusion-outputs/pitch_concat/pitch_concat"
    # elif pitchroll == "roll":
    #     # path = "imu-fusion-outputs/roll_concat/roll_concat"+i+".csv"
    #     path = "imu-fusion-outputs/roll_concat/roll_concat"
    # else:
    #     print("ERROR: Unrecognised input for pressure selector.")
    
    path_pitchroll = "imu-fusion-outputs/pitchroll_concat/"

    csvfiles = glob.glob(path_pitchroll+"**/*.csv",recursive=True) # reading pitch roll data both
    
    # load experimental data matrices from pickle files and create tensors
    # create dataframe for all the tensors, use this in training dataset
    
    data_concat = pd.concat([pd.read_csv(f,delimiter=',',dtype={'GND (euler deg)': float,'Pressure (kPa)': float,'Gyro (deg)': float}) for f in csvfiles])
    # print(data_concat.shape)
    
    if useimu == "no-imu": # without imu
        data_trainX, data_testX, data_trainY, data_testY = train_test_split(data_concat.loc[:,['Pressure (kPa)']], data_concat.loc[:,['GND (euler deg)']], test_size=0.2)

    elif useimu == "use-imu": # with imu
        data_trainX, data_testX, data_trainY, data_testY = train_test_split(data_concat.loc[:,['Pressure (kPa)', 'Gyro (deg)']], data_concat.loc[:,['GND (euler deg)']], test_size=0.2)
    
    else: 
        print("ERROR: Unrecognised input.")
        
    print("TrainX:", data_trainX.shape, "TrainY:", data_trainY.shape, "TestX:", data_testX.shape, "TestY:", data_testY.shape)
    
if __name__ == "__main__":
    
    useimu = sys.argv[1] # save seperate models for pitch and roll
    
    main(useimu)
    # main()
    
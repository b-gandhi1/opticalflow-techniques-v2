import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
# from sklearn import svm
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, roc_curve, RocCurveDisplay, auc)
from sklearn.multioutput import MultiOutputRegressor
# use torch instead of sklearn, numpy, and pandas - more efficient. 
import torch 
import sys
import glob

    
def ml_model(useimu, mcpX_train, mcp_y_train, mcpX_test, mcp_y_test, model):
    
    # gnb = GaussianNB()
    
    mcp_model = MultiOutputRegressor(model) # define model LK map
    
    pipe_mcp = make_pipeline(StandardScaler(), mcp_model)
    
    pipe_mcp.fit(mcpX_train, mcp_y_train)
    
    y_pred_mcp = pipe_mcp.predict(mcpX_test)
    
    # mse_mcp = mean_squared_error(mcp_y_test, y_pred_mcp)
    mse_mcp = np.mean((mcp_y_test - y_pred_mcp)**2)
    rmse_mcp = np.sqrt(mse_mcp)
    mae_mcp = np.mean(np.abs(mcp_y_test - y_pred_mcp))
    
    print("Metric:"+'\t'+"MCP" + '\t' + "MCP+IMU")
    print("Mean Squared Error: ", str(mse_mcp))
    print("Root Mean Squared Error: ", str(rmse_mcp))
    print("Mean Absolute Error: ", str(mae_mcp))
    print("R2 Score: ", str(r2_score(mcp_y_test, y_pred_mcp)))
    
    # Residual Plot
    plt.figure("Residual Plot")
    plt.scatter(y_pred_mcp, mcp_y_test - y_pred_mcp)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    # plt.show()
    
    # Predicted vs Actual Plot
    plt.figure("Predicted vs Actual")
    plt.scatter(mcp_y_test, y_pred_mcp)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual")
    # plt.show()
    
    # cross validations
    scores = cross_val_score(pipe_mcp, mcpX_train, mcp_y_train, cv=5)
    print(f"Cross validation scores mean: {scores.mean():.2f} with standard deviation: {scores.std():.2f}")
    
    plt.show() # finally show all plots
    
    # and then save trained models
    # if useimu == "no-imu":
    #     torch.save(pipe_mcp, 'ml-models/mcp_no-imu_model.pth')
    # elif useimu == "use-imu":
    #     torch.save(pipe_mcp, 'ml-models/mcp_with-imu_model.pth')
    # else:
    #     print("ERROR: Unrecognised input.")

def main(useimu):
    
    path_pitchroll = "imu-fusion-data/pitchroll_concat2/"

    csvfiles = glob.glob(path_pitchroll+"/"+"*.csv",recursive=True) # reading pitch roll data both
    
    # load experimental data matrices from pickle files and create tensors
    # create dataframe for all the tensors, use this in training dataset
    
    pitchroll_df = pd.concat((pd.read_csv(f, delimiter=',', dtype={'Franka Rx': float, 
                                                                   'Franka Ry': float, 
                                                                   'Franka Rz': float,
                                                                   'Pressure (kPa)': float,
                                                                   'IMY Rx': float, 
                                                                   'IMU Ry': float, 
                                                                   'IMU Rz': float, 
                                                                   'LKx': float, 
                                                                   'LKy': float, 
                                                                   'BrtZ': float}) for f in csvfiles), axis='index')
    print(pitchroll_df.shape)
    
    if useimu == "no-imu": # without imu, no imu data used
        data_trainX, data_testX, data_trainY, data_testY = train_test_split(
            pitchroll_df.loc[:,['Pressure (kPa)','LKx','LKy','BrtZ']], 
            pitchroll_df.loc[:,['Franka Rx', 'Franka Ry', 'Franka Rz']], 
            test_size=0.2)

    elif useimu == "use-imu": # with imu, use imu data
        data_trainX, data_testX, data_trainY, data_testY = train_test_split(
            pitchroll_df.loc[:,['Pressure (kPa)','LKx','LKy','BrtZ','IMU Rx', 'IMU Ry']], 
            pitchroll_df.loc[:,['Franka Rx', 'Franka Ry', 'Franka Rz']], 
            test_size=0.2)
    
    else: 
        print("ERROR: Unrecognised input.")
        
    print("TrainX:", data_trainX.shape, "TrainY:", data_trainY.shape, "TestX:", data_testX.shape, "TestY:", data_testY.shape)
    
    # run ml model
    print("Computing ML model: Linear Regression ......")
    ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=LinearRegression())
    
    print("Computing ML model: Decision Tree Regressor ......")
    ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=DecisionTreeRegressor())
    
    print("Computing ML model: Random Forest Regressor ......")
    ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=RandomForestRegressor())
    
    print("Computing ML model: Support Vector Regressor ......")
    ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=SVR())
    
if __name__ == "__main__":
    
    useimu = sys.argv[1] # save seperate models for imu and no-imu
    
    main(useimu)
    # main()
    
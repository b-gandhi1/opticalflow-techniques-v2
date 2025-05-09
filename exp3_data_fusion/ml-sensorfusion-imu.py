import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn import svm
from sklearn.linear_model import LinearRegression, Lasso, LassoLars, ElasticNet, Ridge, BayesianRidge, HuberRegressor, PassiveAggressiveRegressor, RANSACRegressor, SGDRegressor, TheilSenRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
# from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, roc_curve, RocCurveDisplay, auc)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor, BernoulliRBM
import torch
import sys
import glob
from scipy.stats import spearmanr
import wandb
from wandb.sklearn import plot_residuals, plot_outlier_candidates

def ml_model(useimu, mcpX_train, mcp_y_train, mcpX_test, mcp_y_test, model):
    
    # gnb = GaussianNB()
    wandb.init(project='ml_fuse_v1', name=model.__class__.__name__, save_code=True)
    mcp_model = MultiOutputRegressor(model) # define model LK map
    
    pipe_mcp = make_pipeline(StandardScaler(), mcp_model)
    
    pipe_mcp.fit(mcpX_train, mcp_y_train)
    
    y_pred_mcp = pipe_mcp.predict(mcpX_test)
    
    # mse_mcp = mean_squared_error(mcp_y_test, y_pred_mcp)
    mse_mcp = np.mean((mcp_y_test - y_pred_mcp)**2)
    rmse_mcp = np.sqrt(mse_mcp)
    mae_mcp = np.mean(np.abs(mcp_y_test - y_pred_mcp))
    r2_mcp = r2_score(mcp_y_test, y_pred_mcp)
    
    # log metrics to wandb
    wandb.log({"MSE": mse_mcp, "RMSE": rmse_mcp, "MAE": mae_mcp, "R2": r2_mcp})
    print(f"Model: {model.__class__.__name__} MSE: {mse_mcp:.2f} RMSE: {rmse_mcp:.2f} MAE: {mae_mcp:.2f} R2: {r2_mcp:.2f}")
    
    # plot of predictions and ground truth time series: 
    time_ax = np.linspace(0,30,300)
    plt.figure("Predictions and Ground Truth")
    plt.plot(time_ax, mcp_y_test[0:300], label='Ground Truth')
    plt.plot(time_ax, y_pred_mcp[0:300], label='Predictions')
    plt.xlabel("Time (s)")
    plt.ylabel("Values")
    plt.title("Predictions and Ground Truth")
    plt.legend()
    
    # Residual Plot
    plt.figure("Residual Plot")
    plt.scatter(mcp_y_test - y_pred_mcp, bins=100)
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
    wandb.log({"Cross Validation Mean": scores.mean(), "Cross Validation Std": scores.std()})
    print(f"Model: {model.__class__.__name__} Cross validation scores mean: {scores.mean():.2f} with standard deviation: {scores.std():.2f}")
    
    plt.show() # finally show all plots
    
    # wandb plots 
    plot_residuals(y_pred_mcp, mcp_y_test)
    plot_outlier_candidates(y_pred_mcp, mcp_y_test)
    
    # and then save trained models
    # if useimu == "no-imu":
    #     torch.save(pipe_mcp, 'ml-models/mcp_no-imu_model.pth')
    # elif useimu == "use-imu":
    #     torch.save(pipe_mcp, 'ml-models/mcp_with-imu_model.pth')
    # else:
    #     print("ERROR: Unrecognised input.")
    wandb.finish()
def main(useimu, mode):
    
    path_pitchroll = "imu-fusion-data/pitchroll_concat2/"
    
    if mode == "pitch" or "roll": # or "tz":
        read_mode = mode
    else:
        read_mode = "**/"
    csvfiles = glob.glob(path_pitchroll+"/"+read_mode+"*.csv",recursive=True) # reading pitch roll data as specified by mode
    print(csvfiles)
    
    # load experimental data matrices from pickle files and create tensors
    # create dataframe for all the tensors, use this in training dataset
    
    pitchroll_df = pd.concat((pd.read_csv(f, delimiter=',', usecols={'Franka Rx', 'Franka Ry', 
                                                                    'Pressure (kPa)', 
                                                                    'IMU Rx', 'IMU Ry', 
                                                                    'LKx', 'LKy'}, 
                                                                    dtype={'Franka Rx': float, 
                                                                            'Franka Ry': float, 
                                                                            'Pressure (kPa)': float,
                                                                            'IMY Rx': float, 
                                                                            'IMU Ry': float, 
                                                                            'LKx': float, 
                                                                            'LKy': float}) 
                                                                        for f in csvfiles), axis='index')
    print(pitchroll_df.shape)
    
    # tensot (.pt) files: 
    ptfiles = glob.glob(path_pitchroll+"/"+read_mode+"*.pt")
    print(ptfiles) # print the list of selected files, should be x and y data for pitch | roll
    exp_tensor = [torch.load(f) for f in ptfiles]
    # for i, tensor in enumerate(exp_tensor):
    #     print(f"Shape of tensor {i+1}: {tensor.shape}")
    # concatenate tensors: 
    concat_exp_tensor = torch.cat(exp_tensor, dim=0)
    # print("Merged Tensor Shape:", concat_exp_tensor.shape)
    
    # check data loaded: 
    
    # xs = pitchroll_df.loc[300:600+1,['Franka Rx']]
    # ys = pitchroll_df.loc[300:600+1,['Franka Ry']]
    # ts = np.linspace(0,300,len(xs))
    
    # plt.figure()
    
    # plt.subplot(3,1,1)
    # plt.plot(ts, xs, label='franka_Rx')
    # plt.plot(ts, ys, label='franka_Ry')
    # plt.legend()
    # plt.title(mode)

    # plt.subplot(3,1,2)
    # ax1 = plt.gca()   
    # LKx = pitchroll_df.loc[300:600+1,['LKx']]
    # LKy = pitchroll_df.loc[300:600+1,['LKy']]
    # ax1.plot(ts, LKx, 'b-', label='LKx')
    # ax1.set_ylabel('LKx', color='b')
    # ax1.tick_params('y', colors='b')
    # # ax1.legend(loc='upper left')
    # ax2 = ax1.twinx()
    # ax2.plot(ts, LKy, 'g-', label='LKy')
    # ax2.set_ylabel('LKy', color='g')
    # ax2.tick_params('y', colors='g')
    # # ax2.legend(loc='upper right') 
    
    # plt.subplot(3,1,3)
    # imuX = pitchroll_df.loc[300:600+1, ['IMU Rx']]
    # imuY = pitchroll_df.loc[300:600+1, ['IMU Ry']]
    # plt.plot(ts, imuX, label='IMU X')
    # plt.plot(ts, imuY, label='IMU Y')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()
    
    # to_cont = input("Continue? (y/n): ")
    # if to_cont == 'n':
    #     exit()
    # else:
    #     pass
    
    # load tensor files - 
    # tensor_paths = glob.glob("imu-fusion-data/pitchroll_concat2/*.pt")
    # tensor_data_list = [torch.load(f) for f in tensor_paths]
    # tensor_data = torch.cat(tensor_data_list, dim=0)
    
    if useimu == "no-imu" and mode == "pitchroll": # without imu, no imu data used
        experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy']]
        ground_truth = pitchroll_df.loc[:,['Franka Rx', 'Franka Ry']]

    elif useimu == "no-imu" and mode == "pitch":
        experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy']]
        ground_truth = pitchroll_df.loc[:,['Franka Ry']]
    
    elif useimu == "no-imu" and mode == "roll":
        experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy']]
        ground_truth = pitchroll_df.loc[:,['Franka Rx']]

    # elif useimu == "no-imu" and mode == "tz":
    #     experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'BrtZ']]
    #     ground_truth = pitchroll_df.loc[:,['Franka Tz']]
        
    elif useimu == "use-imu" and mode == "pitch":
        experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy', 'IMU Rx']]
        ground_truth = pitchroll_df.loc[:,['Franka Ry']]

    elif useimu == "use-imu" and mode == "roll":
        experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy', 'IMU Ry']]
        ground_truth = pitchroll_df.loc[:,['Franka Rx']]

    elif useimu == "use-imu" and mode == "pitchroll": # with imu, use imu data
        experimental_data = pitchroll_df.loc[:,['Pressure (kPa)', 'LKx', 'LKy', 'IMU Rx', 'IMU Ry']]
        ground_truth = pitchroll_df.loc[:,['Franka Rx', 'Franka Ry']]

    # elif useimu == "use-imu" and mode == 'tz': 
    #     print("ERROR: Invalid Combination, IMU Tz data not available.")
    #     exit()
        
    else: 
        raise ValueError("ERROR: Unrecognised input. Expected inputs are imu= no-imu | use-imu ; ")
        # print("ERROR: Unrecognised input. Expected inputs are imu= no-imu | use-imu ; ")
        # exit()

    data_trainX, data_testX, data_trainY, data_testY = train_test_split(experimental_data, ground_truth, test_size=0.1, shuffle=False) 
    
    # test plots: 
    
    # plt.figure()
    # plt.plot(ts, data_trainX[1:300+3], '-', label='exp dat')
    # plt.plot(ts, data_trainY[1:300+3], '-', label='gnd data')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # to_cont = input("Continue? (y/n): ")
    # if to_cont == 'n':
    #     exit()
    # else:
    #     pass
        
    print("TrainX:", data_trainX.shape, "TrainY:", data_trainY.shape, "TestX:", data_testX.shape, "TestY:", data_testY.shape)
    corr_nonlin, _ = spearmanr(experimental_data, ground_truth, alternative='two-sided',nan_policy='propagate')
    print("Correlation: ", corr_nonlin)
    
    # run ml model
    print("Computing ML model: Linear Regression ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=LinearRegression())
    
    print("Computing ML model: Decision Tree Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=DecisionTreeRegressor())
    
    print("Computing ML model: Random Forest Regressor ......")
    ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=RandomForestRegressor(verbose=True))
    
    print("Computing ML model: Support Vector Regressor ......")
    ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=SVR(kernel='rbf', degree=3, C=5.0, epsilon=0.1, max_iter=1500, verbose=True))
    
    print("Computing ML model: Lasso Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=Lasso())
    
    print("Computing ML model: Lasso Lars Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=make_pipeline(StandardScaler(with_mean=False), LassoLars()))
    
    print("Computing ML model: Elastic Net Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=ElasticNet())
    
    print("Computing ML model: Ridge Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=Ridge())
    
    print("Computing ML model: Bayesian Ridge Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=BayesianRidge())
    
    print("Computing ML model: Huber Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=HuberRegressor())
    
    print("Computing ML model: Passive Aggressive Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=PassiveAggressiveRegressor())
    
    print("Computing ML model: RANSAC Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=RANSACRegressor())
    
    print("Computing ML model: SGD Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=SGDRegressor())
    
    print("Computing ML model: Theil Sen Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=TheilSenRegressor())
    
    print("Computing ML model: Multi-layer Perceptron Regressor ......")
    # ml_model(useimu, data_trainX, data_trainY, data_testX, data_testY, model=MLPRegressor(max_iter=5000,early_stopping=True)) # further fine tune
    
    custom_model = ...
    
if __name__ == "__main__":
    
    useimu = sys.argv[1] # save seperate models for use-imu and no-imu
    
    mode = sys.argv[2] # pitch | roll | pitchroll
    
    try:
        main(useimu, mode)
    except KeyboardInterrupt:
        wandb.finish()
        plt.close('all')
        # print("Keyboard Interrupt. Exiting...")
        raise SystemExit("Keyboard Interrupt. Exiting...")

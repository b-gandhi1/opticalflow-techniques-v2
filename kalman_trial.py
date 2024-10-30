import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, roc_curve, RocCurveDisplay, auc)
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

# const vars
FPS = 10.0 # 10 fps

def kalman_model():
    # Initialize the filter
    kf = KalmanFilter(dim_x=2, dim_z=1)

    kf.F = np.eye(2) # state transition matrix
    kf.H = np.array([[1.,0.]]) # measurement function
    kf.R = np.array([[0.1]]) # measurement noise covariance matrix
    kf.Q = np.eye(2) # process noise covariance matrix

    kf.x = np.zeros((2,1)) # initial state estimate
    kf.P = np.eye(2) # initial state covariance matrix

    for val in training_data: 
        kf.update(val)
        kf.predict()
        
# UKF - https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html 
def fx(x, dt):
# state transition function - predict next state based
# on constant velocity model x = vt + x_0
    F = np.array([[1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]], dtype=float)
    return np.dot(F, x)
def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos]
    return np.array([x[0], x[2]])

def unscented_kf_model():
    dt = 1/FPS 
    # create sigma points to use in the filter. This is standard for Gaussian processes
    points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
    
    kf = kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points) # UnscentedKalmanFilter(dim_x, dim_z, dt, hx, fx, points, sqrt_fn=None, x_mean_fn=None, z_mean_fn=None, residual_x=None, residual_z=None)
    kf.x = np.array([-1., 1., -1., 1]) # initial state
    kf.P *= 0.2 # initial uncertainty
    z_std = 0.1
    kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)

def main():
    data_concat = ... # pitch and roll data both, tensor format 
    
    data_trainX, data_testX, data_trainY, data_testY = train_test_split(data_concat, test_size=0.2)

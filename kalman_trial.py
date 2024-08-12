import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, roc_curve, RocCurveDisplay, auc)
from filterpy.kalman import KalmanFilter

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

def main():
    data_concat = ... # pitch and roll data both, tensor format 
    
    data_trainX, data_testX, data_trainY, data_testY = train_test_split(data_concat, test_size=0.2)

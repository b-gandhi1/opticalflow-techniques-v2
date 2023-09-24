# this file reads the outputs generated from opticalflow.py, and uses them to train a ML model. 

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# load pickle data from outputs folder 
# data_bd_web = pickle.load( open( "outputs/bd_web.pkl", "rb" ) )
data_lk_web = pickle.load( open( "outputs/lk_web.pkl", "rb" ) )
# data_gf_web = pickle.load( open( "outputs/gf_web.pkl", "rb" ) )

# data_bd_fibre = pickle.load( open( "outputs/bd_fibre.pkl", "rb" ) )
data_lk_fibre = pickle.load( open( "outputs/lk_fibre.pkl", "rb" ) )
# data_gf_fibre = pickle.load( open( "outputs/gf_fibre.pkl", "rb" ) )

# load ground truth data from franka, csv file 
fib_gnd_truth = np.loadtxt('outputs/ground_truth_fibre.csv', delimiter=',')
web_gnd_truth = np.loadtxt('outputs/ground_truth_web.csv', delimiter=',')

# frame transformations for franka_ee_pos, with respect to mannequin head origin.  
x_offset, y_offset, z_offset = 20, 0, 68 # mm

# select datasetand scale:
# selectdataset = input('Which dataset? Enter a number(1 - BD, 2 - LK, 3 - GF): ')
whichdata_web = data_lk_web
whichdata_fib = data_lk_fibre

# split data into training, testing and validation sets
web_X_train, web_X_test, web_y_train, web_y_test = train_test_split(whichdata_web, web_gnd_truth, test_size=0.3,random_state=109) # 70% training and 30% test
fib_X_train, fib_X_test, fib_y_train, fib_y_test = train_test_split(whichdata_fib, fib_gnd_truth, test_size=0.3,random_state=109) # 70% training and 30% test

# define model1 SVM:
web_clf_svm = svm.SVC(kernel='rbf', C=1, gamma=0.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None) # RBF Kernel. model 1: SVM webcam
fib_clf_svm = svm.SVC(kernel='rbf', C=1 ,gamma=0.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None) # RBF Kernel. model 1: SVM fibrescope

# define model2 Naive Bayes: 
web_clf_bayes = ...
fib_clf_bayes = ...

# 0 <= gamma <= 1. Closer to one causes overfitting the data. 
# C is the regularization parameter of the error term. The higher the value of C, the more regularization is applied. Maimum value is 1, Minimum value is 0. 

# create pipeline to standarsize data, and fit model
model_select = input('Which model? Enter a number(1 - SVM, 2 - Logistic regression, 3 - Naive Bayes): ')
chosen_model_web = ... # switch case
chosen_model_fib = ... # switch case
pipe_web = make_pipeline(StandardScaler(), chosen_model_web)
pipe_fib = make_pipeline(StandardScaler(), chosen_model_fib)

# fit model: train ML models for each data set
pipe_web.fit(web_X_train, web_y_train)
pipe_fib.fit(fib_X_train, fib_y_train) 

# confusioin matrix on training data 


# test model: test ML models for each data set
web_y_pred_svm = pipe_web.predict(web_X_test)
fib_y_pred_svm = pipe_fib.predict(fib_X_test)

# confusion matrix on testing data


# validation tests for all the models


# comparisons among models


# save results, csv tables and svg images




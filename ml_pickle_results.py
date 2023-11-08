# this file reads the outputs generated from opticalflow.py, and uses them to train a ML model. 

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, roc_curve, auc)
# use torch instead of sklearn, numpy, and pandas - more efficient. 

# load pickle data from outputs folder 
def bd_data_loader():
    data_bd_web = pickle.load( open( "outputs/bd_web.pkl", "rb" ) )
    data_bd_fibre = pickle.load( open( "outputs/bd_fibre.pkl", "rb" ) )
    return data_bd_web, data_bd_fibre
def lk_data_loader():
    data_lk_web = pickle.load( open( "outputs/lk_web.pkl", "rb" ) )
    data_lk_fibre = pickle.load( open( "outputs/lk_fibre.pkl", "rb" ) )
    return data_lk_web, data_lk_fibre

def gf_data_loader():
    data_gf_web = pickle.load( open( "outputs/gf_web.pkl", "rb" ) )
    data_gf_fibre = pickle.load( open( "outputs/gf_fibre.pkl", "rb" ) )
    return data_gf_web, data_gf_fibre

switcher = {
    'bd': bd_data_loader,
    'lk': lk_data_loader,
    'gf': gf_data_loader
}

choose_data_loader = input("Which dataset? Enter an option: 'BD', 'LK', 'GF'): ")
chosen_loader = switcher.get(choose_data_loader)
whichdata_web, whichdata_fib = chosen_loader()

# load ground truth data from franka, csv file 
# fib_gnd_truth = np.loadtxt('outputs/ground_truth_fibre.csv', delimiter=',')
fib_gnd_truth_df = pd.read_csv('outputs/ground_truth_fibre.csv', delimiter=',')
# web_gnd_truth = np.loadtxt('outputs/ground_truth_web.csv', delimiter=',')
web_gnd_truth_df = pd.read_csv('outputs/ground_truth_web.csv', delimiter=',')

# frame transformations for franka_ee_pos, with respect to mannequin head origin.  
x_offset, y_offset, z_offset = 68, 0, 20 # mm, Translation
# and rotation? ...

# split data into training, testing and validation sets
web_X_train, web_X_test, web_y_train, web_y_test = train_test_split(whichdata_web, web_gnd_truth_df, test_size=0.3,random_state=109) # 70% training and 30% test
fib_X_train, fib_X_test, fib_y_train, fib_y_test = train_test_split(whichdata_fib, fib_gnd_truth_df, test_size=0.3,random_state=109) # 70% training and 30% test

# define model SVM:
# web_clf_svm = svm.SVC(kernel='rbf', C=1, gamma=0.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None) # RBF Kernel. model 1: SVM webcam
# fib_clf_svm = svm.SVC(kernel='rbf', C=1 ,gamma=0.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None) # RBF Kernel. model 1: SVM fibrescope
# 0 <= gamma <= 1. Closer to one causes overfitting the data. 
# C is the regularization parameter of the error term. The higher the value of C, the more regularization is applied. Maimum value is 1, Minimum value is 0. 

gnb = GaussianNB() # define ML model1 Gaussian Naive Bayes
# logReg = LogisticRegression() # define ML model2 Logistic Regression

# create pipeline to standarsize data, and fit model
pipe_web = make_pipeline(StandardScaler(), gnb)
pipe_fib = make_pipeline(StandardScaler(), gnb)

# fit model: train ML models for each data set
pipe_web.fit(web_X_train, web_y_train)
pipe_fib.fit(fib_X_train, fib_y_train) 

# test model: test ML models for each data set
web_y_pred = pipe_web.predict(web_X_test)
fib_y_pred = pipe_fib.predict(fib_X_test)

# performance metrics: accuracy and F1 score
web_accuracy = accuracy_score(web_y_pred, web_y_test)
web_f1 = f1_score(web_y_pred, web_y_test, average="weighted")

fib_accurcy = accuracy_score(fib_y_pred, fib_y_test)
fib_f1 = f1_score(fib_y_pred, fib_y_test, average="weighted")

accuracyf1 = '\t Webcam \t Fibrescope' + '\n' + 'Accuracy: ' + str(web_accuracy) + '\t' + str(fib_accurcy) + '\n' + 'F1 score: ' + str(web_f1) + '\t' + str(fib_f1)
print('\t Webcam \t Fibrescope')
print('Accuracy: ' + str(web_accuracy) + '\t' + str(fib_accurcy))
print('F1 score: ' + str(web_f1) + '\t' + str(fib_f1))

# confusion matrix on testing data


# validation tests for all the models


# comparisons among models


# save results, csv tables and svg images 
np.savetxt('outputs/results.txt', accuracyf1, delimiter='\n')

# save the models
pickle.dump(pipe_web, open( "outputs/web_GNB.sav", "wb" ) )
pickle.dump(pipe_fib, open( "outputs/fib_GNB.sav", "wb" ) )

# To load models and get results, use: --------------------------------
# load_model_web = pickle.load( open( "outputs/web_GNB.sav", "rb" ) )
# result_web = load_model_web.score(web_X_test, web_y_test)
# print('webcam result: ',result)
# load_model_fib = pickle.load( open( "outputs/fib_GNB.sav", "rb" ) )
# result_fib = load_model_fib.score(fib_X_test, fib_y_test)
# print('fibrescope result: 'result)




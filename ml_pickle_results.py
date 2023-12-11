# this file reads the outputs generated from opticalflow.py, and uses them to train a ML model. 

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
# from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, roc_curve, RocCurveDisplay, auc)
# use torch instead of sklearn, numpy, and pandas - more efficient. 
import torch 
import sys

# load pickle data from outputs folder 
def bd_data_loader_gray():
    bd_web = pickle.load( open( "OF_outputs/BD_gray_web1_2023-12-06_16-57-23.pkl", "rb" ) )
    bd_fibre = pickle.load( open( "OF_outputs/BD_gray_fib1_2023-12-07_12-30-19.pkl", "rb" ) )
    
    max_len_web = max(len(d['data']) for d in bd_web)
    padded_web = [np.pad(d['data'], (0, max_len_web - len(d['data'])), mode='constant') for d in bd_web]
    data_bd_web = np.asarray(padded_web)
    
    # for i in range(len(data_bd_web)): # test shapes to identify inhomogenity
    #     print(data_bd_web[i].shape)
        
    max_len_fibre = max(len(d['data']) for d in bd_fibre)
    padded_fibre = [np.pad(d['data'], (0, max_len_fibre - len(d['data'])), mode='constant') for d in bd_fibre]
    data_bd_fibre = np.asarray(padded_fibre)
        
    return data_bd_web, data_bd_fibre

def bd_data_loader_binary():
    bd_web = pickle.load( open( "OF_outputs/BD_binary_web1_2023-12-06_16-49-54.pkl", "rb" ) )
    bd_fibre = pickle.load( open( "OF_outputs/BD_binary_fib1_2023-12-06_17-47-48.pkl", "rb" ) )
    
    data_bd_web = np.asarray([d['data'] for d in bd_web])
    data_bd_fibre = np.asarray([d['data'] for d in bd_fibre])
    
    return data_bd_web, data_bd_fibre

def lk_data_loader_gray():
    lk_web = pickle.load( open( "OF_outputs/LK_gray_web1_2023-12-04_17-02-48.pkl", "rb" ) )
    lk_fibre = pickle.load( open( "OF_outputs/LK_bright_fib1_2023-11-29_16-44-01.pkl", "rb" ) )
    
    data_lk_web = np.asarray([d['data'] for d in lk_web])
    data_lk_fibre = np.asarray([d['data'] for d in lk_fibre])
    
    return data_lk_web, data_lk_fibre

def lk_data_loader_binary():
    lk_web = pickle.load( open( "OF_outputs/LK_binary_web1_2023-12-06_17-56-36.pkl", "rb" ) )
    lk_fibre = pickle.load( open( "OF_outputs/LK_binary_fib1_2023-11-29_16-37-46.pkl", "rb" ) )
    
    data_lk_web = np.asarray([d['data'] for d in lk_web])
    data_lk_fibre = np.asarray([d['data'] for d in lk_fibre])
    
    return data_lk_web, data_lk_fibre

# def gf_data_loader():
#     data_gf_web = pickle.load( open( "OF_outputs/gf_web.pkl", "rb" ) )
#     data_gf_fibre = pickle.load( open( "OF_outputs/gf_fibre.pkl", "rb" ) )
#     return data_gf_web, data_gf_fibre

# gray, binary 
def main(whichmodel):
    switcher = {
        'bd_bin': bd_data_loader_binary,
        'bd_gray': bd_data_loader_gray,
        'lk_bin': lk_data_loader_binary,
        'lk_gray': lk_data_loader_gray
        # 'gf': gf_data_loader
    }

    choose_data_loader = whichmodel
    chosen_loader = switcher.get(choose_data_loader)
    whichdata_web, whichdata_fib = chosen_loader()

    print('Shape web: ', np.shape(whichdata_web))
    print('Shape fib: ', np.shape(whichdata_fib))
    
    # load ground truth data from franka, csv file 
    # fib_gnd_truth = np.loadtxt('outputs/ground_truth_fibre.csv', delimiter=',')
    fib_gnd_truth_df = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope1-20-Nov-2023--14-06-58.csv', delimiter=',')
    fib_gnd_truth_df = fib_gnd_truth_df[1:] # remove first data point to match sizes
    # web_gnd_truth = np.loadtxt('outputs/ground_truth_web.csv', delimiter=',')
    web_gnd_truth_df = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/webcam1-20-Nov-2023--15-56-11.csv', delimiter=',')
    web_gnd_truth_df = web_gnd_truth_df[1:] # remove first data point to match sizes
    
    print('Shape web ground truth: ', np.shape(web_gnd_truth_df))
    print('Shape fib ground truth: ', np.shape(fib_gnd_truth_df))
    
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
    pipe_web.fit(web_X_train, web_y_train) # ERROR HERE with dict file format. 
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
    plt.figure('Webcam Confusion Matrix')
    ConfusionMatrixDisplay.from_predictions(web_y_test, web_y_pred)
    # plt.show()
    plt.figure('Fibrescope Confusion Matrix')
    ConfusionMatrixDisplay.from_predictions(fib_y_test, fib_y_pred)
    # plt.show()
    
    print('Webcam classification report: ', '\n', classification_report(web_y_test, web_y_pred))
    print('==========================================================')
    print('Fibrescope classification report: ', '\n', classification_report(fib_y_test, fib_y_pred))
    
    plt.figure('ROC curves')
    fpr_web, tpr_web, _ = roc_curve(web_y_test, web_y_pred)
    # plt.figure('Webcam ROC, AUC = '+ str(auc(fpr_web, tpr_web)))
    roc_disp_web = RocCurveDisplay(fpr=fpr_web, tpr=tpr_web).plot('Webcam (gray), AUC='+str(auc(fpr_web, tpr_web)))
    
    fpr_fib, tpr_fib, _ = roc_curve(fib_y_test, fib_y_pred)
    # plt.figure('Fibrescope ROC, AUC = '+ str(auc(fpr_fib, tpr_fib)))
    roc_disp_fib = RocCurveDisplay(fpr=fpr_fib, tpr=tpr_fib, ax = roc_disp_web.ax_).plot(label = 'Fibrescope (gray), AUC='+str(auc(fpr_fib, tpr_fib)))
    
    plt.axis("square")
    plt.legend()
    # plt.legend([roc_disp_web, roc_disp_fib], ['Webcam (gray), AUC='+str(auc(fpr_web, tpr_web)), 'Fibrescope (gray), AUC='+str(auc(fpr_fib, tpr_fib))])
    plt.show()
    
    # validation tests for all the models
    scores_web = cross_val_score(gnb, web_X_test, web_y_test, cv=3)
    scores_fib = cross_val_score(gnb, fib_X_test, fib_y_test, cv=3)
    
    print('Webcam cross-validation accuracy: %0.2f  with standard deviaiton %0.2f' % (scores_web.mean(), scores_web.std())) 
    print('Fibrescope cross-validation accuracy: %0.2f  with standard deviaiton %0.2f' % (scores_fib.mean(), scores_fib.std()))
    
    # save results, csv tables and svg images 
    np.savetxt('outputs/results.txt', accuracyf1, delimiter='\n')
    # CONTINUE HERE....
    
    # save models using pytorch
    torch.save(pipe_web, 'ML_results/web_GNB.pt')
    torch.save(pipe_fib, 'ML_results/fib_GNB.pt')

    # To load models and get results, use: --------------------------------
    # load_model_web = pickle.load( open( "outputs/web_GNB.sav", "rb" ) )
    # result_web = load_model_web.score(web_X_test, web_y_test)
    # print('webcam result: ',result)
    # load_model_fib = pickle.load( open( "outputs/fib_GNB.sav", "rb" ) )
    # result_fib = load_model_fib.score(fib_X_test, fib_y_test)
    # print('fibrescope result: ',result)

if __name__ == "__main__":
    whichmodel = sys.argv[1]
    main(whichmodel)


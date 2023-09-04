#!/usr/bin/python
# -*- coding: utf-8 -*- 

# this file reads the outputs generated from opticalflow.py, and uses them to train a ML model. 

import pickle
import numpy as np
import matplotlib.pyplot as plt

# load pickle data from outputs folder 
data_bd_web = pickle.load( open( "outputs/bd_web_trial.pkl", "rb" ) )
data_lk_web_trial = pickle.load( open( "outputs/lk_web_trial.pkl", "rb" ) )

# load ground truth data


# split data into training, testing and validation sets


# train ML models for each data set


# test ML models for each data set


# validation tests for all the models


# comparisons among models


# save data




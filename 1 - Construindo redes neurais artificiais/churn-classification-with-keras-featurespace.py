# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 07:51:33 2023

@author: carlo
"""

# DATA PREPARATION

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import FeatureSpace
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.5f}'.format

# Importing data
dataframe = pd.read_csv('./data/Churn_Modelling.csv')

# Split independent and dependent data
X = dataframe.iloc[:, 3:13]
y = dataframe.iloc[:, 13]

# Separating training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Generating tf.data.Dataset objects for each dataframe
def dataframe_to_tf_dataset(dataframe, labels):
    dataframe = dataframe.copy()
    labels = labels.copy()
    # .from_tensor_slices() => Return the objects of sliced elements
    '''Each Dataset yields a tuple (input, target) where input is a dictionary of 
    features and target is the value 0 or 1'''
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_tf_dataset(X_train, y_train)
test_ds = dataframe_to_tf_dataset( X_test, y_test)

# Batching the datasets
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch
train_ds = train_ds.batch(32)
test_ds = test_ds.batch(32)

# Configuring a FeatureSpace




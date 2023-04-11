# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 07:51:33 2023

@author: carlo
"""

# LOADING MODULES

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

# DATA PREPARATION

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
feature_space = FeatureSpace(
    features={
        "Tenure": FeatureSpace.float_normalized(),
        "Age": FeatureSpace.float_discretized(num_bins=20),
        "CreditScore": FeatureSpace.float_normalized(),
        "Balance": FeatureSpace.float_normalized(),
        "NumOfProducts": FeatureSpace.float_normalized(),
        "EstimatedSalary": FeatureSpace.float_normalized(),
        "Geography": FeatureSpace.string_categorical(),
        "Gender": FeatureSpace.string_categorical(),
        "HasCrCard": FeatureSpace.integer_categorical(),
        "IsActiveMember": FeatureSpace.integer_categorical(),
    },
    crosses=[
        FeatureSpace.cross(feature_names=("Age", "Gender"), crossing_dim=64),
    ],
    output_mode="concat",
)

#Applying FeatureSpace
train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

# Creating a training and validation dataset of preprocessed batches
preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = test_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

# BUILDING A MODEL
dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 07:57:20 2023

@author: carlo
"""

# Libraries

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import  make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import MeanSquaredError
import tensorflow.keras.callbacks as kr_callbacks
import seaborn as sn
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.5f}'.format

# Importing data
housing = fetch_california_housing()

# DATA PREPARATION

# Split independent and dependent data
X = housing.data
y = housing.target

features_names = housing.feature_names
target_name = housing.target_names

# Separating training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature scaling
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.fit_transform(X_test)

# MODELING NEURAL NETWORK

regressor = Sequential()

# Creating layers

regressor.add(Dense(units = 30, activation = 'relu', input_dim = X_train.shape[1], input_shape = X_train.shape[1:]))
regressor.add(Dense(units = 1))

# Compiling model

regressor.compile(optimizer='sgd', loss='mse')

# Training model

early_stopping = kr_callbacks.EarlyStopping(monitor = 'val_loss', patience=3)
history = regressor.fit(X_train, y_train, batch_size=32, epochs=500, validation_data = (X_test, y_test), callbacks=[early_stopping])

y_pred = regressor.predict(X_test)

#Exporting learning curves
pd.DataFrame(history.history).plot(figsize=(10,7))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.savefig("imgs/regression/learning-curves-30-neurons.png")
plt.show();

#Exporting model
regressor.save('./saved-models/regressor')
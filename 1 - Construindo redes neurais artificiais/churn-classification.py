# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:46:44 2023

@author: Carlos Alberto Silva Júnior
"""



# DATA PREPARATION# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import  make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sn
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.5f}'.format

# Importing data
dataset = pd.read_csv('Churn_Modelling.csv')

# Split independent and dependent data
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoders - Categorical to numerical
# Variable Geography
labelEncoder_x_1 = LabelEncoder()
X[:,1] = labelEncoder_x_1.fit_transform(X[:,1])

# Variable Gender
labelEncoder_x_2 = LabelEncoder()
X[:,2] = labelEncoder_x_1.fit_transform(X[:,2])

# Creating dummy variables for Gography values
oneHotEncoder = make_column_transformer((OneHotEncoder(categories = 'auto', sparse = False), [1]), remainder = 'passthrough')
X = oneHotEncoder.fit_transform(X)

#Resolving problem with multicollinearity of dummies variables excluding one collumn from dummies variables created for the feature Geography.
# https://www.algosome.com/articles/dummy-variable-trap-regression.html
X = X[:, 1:]

# Separating training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature scaling
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.fit_transform(X_test)

# MODELING NEURAL NETWORK

classifier = Sequential()

# Creating layers

# Dense layer, with 6 neurons, with uniform initialization and 6 neurons in the input layer

'''Is possible to use the mean between the number of neurons 
in the input layer with the number of neurons in then output
layer to configure the number of neurons in the hidden layers.
This is one way to configure the quantity of neurons in the 
hiddens layers.'''

classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling model

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training model

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)

# Apply threshold
y_pred = (y_pred > 0.5)

# Evaluating the model

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (10,7))
plt.title('Confusion matrix')
cfm_plot = sn.heatmap(cm, annot=True, fmt='g')
cfm_plot.figure.savefig("cfm-10-neurons.png")


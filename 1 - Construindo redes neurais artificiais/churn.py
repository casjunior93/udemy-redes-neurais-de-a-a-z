# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:46:44 2023

@author: Carlos Alberto Silva Júnior
"""

# Importando bibliotecas
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import  make_column_transformer

#importando dados
dataset = pd.read_csv('Churn_Modelling.csv')

# Separando dados independentes e dependente
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoders - Categóricos para numéricos
labelEncoder_x_1 = LabelEncoder()
X[:,1] = labelEncoder_x_1.fit_transform(X[:,1])

labelEncoder_x_2 = LabelEncoder()
X[:,2] = labelEncoder_x_1.fit_transform(X[:,2])
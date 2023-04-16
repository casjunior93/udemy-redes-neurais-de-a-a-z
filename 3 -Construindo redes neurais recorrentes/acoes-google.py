# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 22:06:00 2023

@author: Carlos
"""

#Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Importar os dados
dataset_treino = pd.read_csv('./dados/Google_Stock_Price_Train.csv')
training_set = dataset_treino.iloc[:, 1:2].values

# Feature scaling
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Criando janelas de 60 dias

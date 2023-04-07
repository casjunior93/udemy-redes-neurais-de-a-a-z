# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:46:44 2023

@author: Carlos Alberto Silva Júnior
"""

# Importando bibliotecas
import pandas as pd

#importando dados
dataset = pd.read_csv('Churn_Modelling.csv')

# Separando dados independentes e dependente
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
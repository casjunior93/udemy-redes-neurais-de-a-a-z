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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#Importar os dados
dataset_treino = pd.read_csv('./dados/Google_Stock_Price_Train.csv')
training_set = dataset_treino.iloc[:, 1:2].values

# Feature scaling
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Criando janelas de 60 dias
X_train = []
y_train = []

for i in range(60,1258):
    X_train.append(training_set_scaled[i-60: i,0])
    y_train.append(training_set_scaled[i,0])

# Transformar em np.array
X_train, y_train = np.array(X_train), np.array(y_train)

#Converter para o formato que o Keras precisa
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Construindo a rede neural recorrente
regressor = Sequential()
#1
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
#2
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
#3
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
#4
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
#Saída
regressor.add(Dense(units=1))

# Compilando o modelo
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Treinando a rede
regressor.fit(X_train, y_train, epochs=50, batch_size=32)

#Importar os dados de teste
dataset_teste = pd.read_csv('./dados/Google_Stock_Price_Test.csv')

real_stock_price = dataset_teste.iloc[:, 1:2].values

# Concatenando dados de treino e os dados de teste originais
dataset_total = pd.concat((dataset_treino['Open'], dataset_teste['Open']), axis = 0)

# 60 dias anteriores à janeiro
inputs = dataset_total[len(dataset_total) - len(dataset_teste) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []

#O dataset de teste tem apenas 20 dias úteis
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Dados Reais de Ações do Google')
plt.plot(predicted_stock_price, color = 'blue', label = 'Dados Previstos de Ações do Google')
plt.title('Previsão de Preços de Ações')
plt.xlabel('Tempo')
plt.ylabel('Preços de Ações do Google')
plt.legend()
plt.savefig("./img/precos.png")
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 07:29:08 2023

@author: carlo
"""

# Importando bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Criando o classificador
classifier = Sequential()

# Parâmetros

# Número de filtros, detectores de características
filtros = 32
# Kernel
kernel = (3, 3)
# Formato do input
input_shape = (64, 64, 3)

classifier.add(
    Conv2D(filtros, kernel, input_shape=input_shape, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(
    Conv2D(filtros, kernel, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

# Como experimento, definimos 128 neurônios
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compilando o modelo
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Pré-processamento das imagens
train_dataimg = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_dataimg = ImageDataGenerator(rescale=1./255)

# Carregar os dados de treino
train_set = train_dataimg.flow_from_directory(
    'data/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# Carregando dados de teste
test_set = test_dataimg.flow_from_directory(
    'data/test_set',  target_size=(64, 64), batch_size=32, class_mode='binary')

# Treinando a rede neural
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, mode='min')
history = classifier.fit(train_set, steps_per_epoch=8000//32, epochs=500,
                         validation_data=test_set, validation_steps=2000//32, callbacks=[early_stopping])

# Exporting learning curves
pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.savefig("imgs/learning-curves-128-neurons.png")

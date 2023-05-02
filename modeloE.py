# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:30:22 2023

@author: Emily
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:27:17 2023

@author: Emily
"""

#Modelo E

import tensorflow as tf
import os
import pandas as pd
import numpy as np


df = pd.read_csv('jurumirimatt.csv', sep=';', parse_dates=['Data'], index_col='Data') #, dtype=(dateparse))
df = df.replace({',': '.'}, regex=True)

df['Cota'] = df['Cota'].str.replace(',','.').astype(float)
df['Afluente'] = df['Afluente'].str.replace(',','.').astype(float)
df['Defluente'] = df['Defluente'].str.replace(',','.').astype(float)
print(df.info())
print(df.head())

#df = df[5::6]
#print(df)

#df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
#print(df[:26])

afluente = df['Afluente']
#afluente.plot()

def df_to_X_y(df, window_size=5): #window_size referente a quantos dados vamos utilizar
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+5]]
        X.append(row)
        label = df_as_np[i+5]
        y.append(label)
    return np.array(X), np.array(y)

window_size = 5
X, y = df_to_X_y(afluente, window_size)
print(X.shape, y.shape)

X_train, y_train = X[:8500], y[:8500] #quantidade de linhas
X_val, y_val = X[8500:10500], y[8500:10500]
X_test, y_test = X[10500:], y[10500:]


print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model1 = Sequential()
model1.add(InputLayer((5, 1)))
model1.add(LSTM(64))
model1.add(Dense(8,'relu'))
model1.add(Dense(1,'linear'))

print(model1.summary())

cp = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])

model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[cp]) #epoch quantas vezes vai rodar

from tensorflow.keras.models import load_model

model1 = load_model('model1/')

train_predictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
print(train_results)

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'

plt.plot(train_results['Train Predictions'][:100], label='Previsão')
plt.plot(train_results['Actuals'][:100], label='Valores Reais')
plt.title('Gráfico de Previsão de Treinamento - Modelo E ')
plt.xlabel('Período de Tempo (dias)')
plt.ylabel('Vazão alfuente (m³/s)')
plt.legend(loc='upper right', facecolor='none')
plt.grid()
plt.show()

plt.plot(train_results['Train Predictions'][:1000], label='Previsão')
plt.plot(train_results['Actuals'][:1000], label='Valores Reais')
plt.title('Gráfico de Previsão de Treinamento - Modelo E')
plt.xlabel('Período de Tempo (dias)')
plt.ylabel('Vazão alfuente (m³/s)')
plt.legend(loc='upper right', facecolor='none')
plt.grid()
plt.show()

val_predictions = model1.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
print(val_results)

plt.plot(val_results['Val Predictions'][:100], label='Previsão')
plt.plot(val_results['Actuals'][:100], label='Real')
plt.title('Gráfico de Previsão de Validação - Modelo E')
plt.xlabel('Período de Tempo (dias)')
plt.ylabel('Vazão alfuente (m³/s)')
plt.legend(loc='upper left', facecolor='none')
plt.grid()
plt.show()

test_predictions = model1.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
print(test_results)

plt.plot(test_results['Test Predictions'][:100], label='Previsão')
plt.plot(test_results['Actuals'][:100], label='Real')
plt.title('Gráfico de Previsão de Teste - Modelo E')
plt.xlabel('Período de Tempo (dias)')
plt.ylabel('Vazão alfuente (m³/s)')
plt.legend(loc='upper left', facecolor='none')
plt.grid()
plt.show()

train_results.to_csv('modeloE_train_results.csv', index=False, sep=";", float_format='%.2f')
val_results.to_csv('modeloE_val_results.csv', index=False, sep=";", float_format='%.2f')

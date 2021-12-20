#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
import random

fs = 32000 #sampling rate
Ts = 1/fs #time duration
#t=0.1 
freqList = [250,500, 750 ,1000, 1500, 2000, 3000, 4000, 6000] 
#n = np.linspace(0,1,2048) #no of samples
n = np.arange(0,2048,1) #no of samples
print("n = ",n)
noise = np.random.normal(2048) 
phase = 2* np.pi *np.random.random(2048)
for freq in freqList:
    y = np.sin(2* np.pi * freq *n * Ts) + noise + phase
    plt.figure(figsize = (8, 6))
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.plot(n,y)
    plt.savefig('Sinewave.png'.format(freq))
    plt.show()


# In[21]:


import pandas as pd
import seaborn as sns
from math import sqrt
import math
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense,Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import LSTM, MaxPooling1D, Conv1D, Flatten, LeakyReLU, Activation, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import datetime
import matplotlib.pyplot as plt
from pandas import Series
from numpy.random import randn
import os
import random as rn
import numpy as np
import time
import tensorflow as tf


# In[22]:


#using the MachineLearningMastery formula for splitting up the dataset to predictors and target
def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    """
    A method to create X and Y matrix from a time series array for the training of 
    deep learning models 
    """
    # Extracting the number of features that are passed from the array 
    n_features = ts.shape[1]
    
    # Creating placeholder lists
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    # Reshaping the X array to an RNN input shape 
    X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y


# In[23]:


# Number of lags (hours back) to use for models
lag = 256
# Steps in future to forecast 
n_ahead = 64
# ration of observations in training from total series
train_share = 0.7
val_share = 0.8
# training epochs
epochs = 10
# Batch size , which is the number of samples of lags
batch_size = 1
# Learning rate
lr = 0.001


# In[24]:


data = y.reshape(-1,1)


# In[25]:


data.shape


# In[26]:


#Scaling data between 0 and 1
scaler = MinMaxScaler()
scaler.fit(data)
ts_scaled = scaler.transform(data)


# In[27]:


# Creating the X and Y for training
X, Y = create_X_Y(ts_scaled, lag=lag, n_ahead=n_ahead)


# In[28]:


# Spliting into train and test sets 
Xtrain, Ytrain = X[0:int(X.shape[0] * train_share)], Y[0:int(X.shape[0] * train_share)]
Xtest, Ytest = X[int(X.shape[0] * val_share):], Y[int(X.shape[0] * val_share):]
Xval, Yval = X[int(X.shape[0] * train_share):int(X.shape[0] * val_share)], Y[int(X.shape[0] * train_share):int(X.shape[0] * val_share)]


# In[29]:


Xtrain.shape


# In[30]:


Xtest.shape


# In[31]:


Xval.shape


# In[32]:


#Neural Network Model configuration
model = Sequential()

model.add(LSTM(32, activation='relu', input_shape=(Xtrain.shape[1], Xtrain.shape[2]), return_sequences=True))
#model.add(CuDNNLSTM(32, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), return_sequences=True))
model.add(LSTM(16, activation='relu', return_sequences=False))
#model.add(CuDNNLSTM(16, return_sequences=False))
model.add(Dense(64))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, min_delta=0.001)
model.summary()

model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss='mae', metrics='mae')


# In[33]:


#Train model on train data
history = model.fit(Xtrain, Ytrain,epochs=epochs, validation_data=(Xval, Yval), shuffle=False, callbacks=[early_stopping])
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label = 'Validation loss')
plt.legend()


# In[34]:


yhat = model.predict(Xtest)
yhat.shape


# In[35]:


pred_n_ahead = pd.DataFrame(yhat[0])
actual_n_ahead = pd.DataFrame(Ytest[0])


# In[36]:


Ytest.shape


# In[37]:


#plot n_steps ahead for predicted and actual data
plt.figure(figsize=(15, 8))
plt.plot(pred_n_ahead, color='C0', marker='o', label='Predicted')
plt.plot(actual_n_ahead, color='C1', marker='o', label='Actual', alpha=0.6)
plt.title('Predicted vs Actual Power')
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.legend()
plt.show


# In[38]:


#evaluation metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    return({'mape':mape, 'me':me, 'mae': mae,  'rmse':rmse, 
            'corr':corr})

forecast_accuracy(yhat,Ytest)


# In[ ]:





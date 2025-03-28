import os
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mat73
import h5py
import scipy.io as scipy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
print(os.getcwd())
os.chdir("/home/armin/Desktop/paper3_python/paper3_py")


data = mat73.loadmat('data.mat')
data = scipy.loadmat('data.mat')

data = data["data"]
X = np.delete(data, [6,9,11], axis=1)
Y = data[:,4]



MM = MinMaxScaler()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle=False)
X_train_n = MM.fit_transform(X_train)
X_test_n = MM.transform(X_test)

SC = StandardScaler()
X_train_n1 = SC.fit_transform(X_train)
X_test_n1 = SC.transform(X_test)


# Feature engineering
variance = X_train_n.var(axis=0)
print([variance > 0.01])

features = np.ones([1,X_train_n.shape[1]])
selected_features = (variance.reshape(1,-1) > 0.01).astype(int)
fig, ax = plt.subplots()
plt.plot(X_train_n)
plt.plot(X_train_n1)
plt.show()

# Data shape Designing:
# Y_train_CNN = Y_train.reshape(-1,1)
num_features = X_train_n1.shape[1]
window_size = 30
X_CNN = np.zeros([X_train_n1.shape[0]-window_size,window_size,num_features])
Y_CNN = np.zeros([X_train_n1.shape[0]-window_size,window_size,1])

X_CNN_test = np.zeros([X_test_n1.shape[0]-window_size,window_size,num_features])
Y_CNN_test = np.zeros([X_test_n1.shape[0]-window_size,window_size,1])

for i in range(X_train_n1.shape[0]-window_size):
    X_CNN[i,0:window_size,:] = X_train_n1[i:i+window_size,:]

for i in range(X_train_n1.shape[0]-2*window_size):   
    Y_CNN[i,:,0] = Y_train[i+window_size:i+window_size*2]

for i in range(X_test_n1.shape[0]-window_size):
    X_CNN_test[i,0:window_size,:] = X_test_n1[i:i+window_size,:]

for i in range(X_test_n1.shape[0]-2*window_size):   
    Y_CNN_test[i,:,0] = Y_test[i+window_size:i+window_size*2]


# Model designing
import tensorflow
from tensorflow import keras
from keras import Model
from keras.layers import Dense, LSTM, GRU, Conv1D, Input, MaxPooling1D, Dropout, Flatten, TimeDistributed
from keras.optimizers import Adam, AdamW, SGD
from keras.regularizers import l1




def NN3(num_hiddens = 64,window_size_in=30, window_size=30, alpha=0.001, l1_coef = 0.01):
    inputs  = Input(shape=(window_size_in,1))
    in1     = Dense(num_hiddens, activation = 'relu', kernel_regularizer=l1(l1_coef))(inputs)
    in2     = Dense(num_hiddens, activation = 'relu', kernel_regularizer=l1(l1_coef))(in1)
    in3     = Dense(num_hiddens, activation = 'relu', kernel_regularizer=l1(l1_coef))(in2)
    output  = TimeDistributed(window_size, activation = 'linear')(in3)

    model = Model(inputs,output)
    model.compile(optimizer = AdamW(learning_rate = alpha), loss="mse", metrics=['mae'])
    return model

def LSTM3(num_hiddens = 64,window_size_in=30, window_size=30, alpha=0.001, l1_coef = 0.01):
    inputs = Input(shape=(window_size_in,1))
    x = LSTM(num_hiddens, return_sequences = True, kernel_regularizer=l1(l1_coef))(inputs)
    x = LSTM(num_hiddens, return_sequences = True, kernel_regularizer=l1(l1_coef))(x)
    x = LSTM(num_hiddens, return_sequences = True, kernel_regularizer=l1(l1_coef))(x)
    x = Flatten(x)
    output  = TimeDistributed(window_size, activation = 'linear')(x)

    model = Model(inputs,output)
    model.compile(optimizer = AdamW(learning_rate = alpha), loss="mse", metrics=['mae'])
    return model

def CNN3(num_kernels1 = 64, num_kernels2 = 64,num_kernels3 = 64,kernel_size=5,window_size_in=30, window_size=30, alpha=0.001, l1_coef = 0.01):
    inputs = Input(shape=(window_size_in,1))
    x = Conv1D(num_kernels1, kernel_size, activation='relu', padding = 'same', kernel_regularizer=l1(l1_coef))(inputs)
    x = MaxPooling1D(x)
    x = Conv1D(num_kernels2, kernel_size, activation='relu', padding = 'same', kernel_regularizer=l1(l1_coef))(x)
    x = MaxPooling1D(x)
    x = Conv1D(num_kernels3, kernel_size, activation='relu', padding = 'same', kernel_regularizer=l1(l1_coef))(x)
    x = MaxPooling1D(x)
    x = Flatten(x)
    output  = TimeDistributed(window_size, activation = 'linear')(x)

    model = Model(inputs,output)
    model.compile(optimizer = AdamW(learning_rate = alpha), loss="mse", metrics=['mae'])
    return model


def CNN2_LSTM(num_hiddens=64, num_kernels1 = 64, num_kernels2 = 64,kernel_size=5,window_size_in=30, window_size=30, alpha=0.001, l1_coef = 0.01):
    inputs = Input(shape=(window_size_in,1))
    x = Conv1D(num_kernels1, kernel_size, activation='relu', padding = 'same', kernel_regularizer=l1(l1_coef))(inputs)
    x = MaxPooling1D(x)
    x = Conv1D(num_kernels2, kernel_size, activation='relu', padding = 'same', kernel_regularizer=l1(l1_coef))(x)
    x = MaxPooling1D(x)
    x = LSTM(num_hiddens, return_sequences = True, kernel_regularizer=l1(l1_coef))(x)
    x = Flatten(x)
    output  = TimeDistributed(window_size, activation = 'linear')(x)

    model = Model(inputs,output)
    model.compile(optimizer = AdamW(learning_rate = alpha), loss="mse", metrics=['mae'])
    return model

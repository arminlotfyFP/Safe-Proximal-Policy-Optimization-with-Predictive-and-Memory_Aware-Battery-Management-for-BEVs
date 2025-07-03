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


<<<<<<< HEAD
data = mat73.loadmat('data.mat')
=======
# data = mat73.loadmat('data.mat')
>>>>>>> aabf279 (All files)
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
<<<<<<< HEAD
variance = X_train_n.var(axis=0)
print([variance > 0.01])

features = np.ones([1,X_train_n.shape[1]])
selected_features = (variance.reshape(1,-1) > 0.01).astype(int)
fig, ax = plt.subplots()
plt.plot(X_train_n)
plt.plot(X_train_n1)
plt.show()
=======
# variance = X_train_n.var(axis=0)
# print([variance > 0.01])

features = np.ones([1,X_train_n.shape[1]])
selected_features = (variance.reshape(1,-1) > 0.01).astype(int)
# fig, ax = plt.subplots()
# plt.plot(X_train_n)
# plt.plot(X_train_n1)
# plt.show()
>>>>>>> aabf279 (All files)

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




<<<<<<< HEAD
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

=======
def NN3(num_hiddens1=64,num_hiddens2=64,num_hiddens3=64, window_size_in=30, window_size=30, alpha=0.001, l1_coef=0.01):
    inputs = Input(shape=(330, ))
    x = Dense(num_hiddens1, activation='relu', kernel_regularizer=l1(l1_coef))(inputs)
    x = Dense(num_hiddens2, activation='relu', kernel_regularizer=l1(l1_coef))(x)
    x = Dense(num_hiddens3, activation='relu', kernel_regularizer=l1(l1_coef))(x)
    output = Dense(window_size, activation='linear')(x)

    model = Model(inputs, output)
    model.compile(optimizer=AdamW(learning_rate=alpha), loss='mse', metrics=['mae'])
    return model


from keras.models import Model
from keras.layers import Input, LSTM, Dense, Flatten
from keras.regularizers import l1
from keras.optimizers import Adam
from tensorflow_addons.optimizers import AdamW  # Ensure you have this installed

def LSTM3(num_hiddens1=64,num_hiddens2=64,num_hiddens3=64, window_size_in=30, window_size=30, alpha=0.001, l1_coef=0.01):
    inputs = Input(shape=(window_size_in, 11))
    x = LSTM(num_hiddens1, return_sequences=True, kernel_regularizer=l1(l1_coef))(inputs)
    x = LSTM(num_hiddens2, return_sequences=True, kernel_regularizer=l1(l1_coef))(x)
    x = LSTM(num_hiddens3, return_sequences=True, kernel_regularizer=l1(l1_coef))(x)
    x = Flatten()(x)
    output = Dense(window_size, activation='linear')(x)
    model = Model(inputs, output)
    model.compile(optimizer=AdamW(learning_rate=alpha), loss='mse', metrics=['mae'])

    return model


>>>>>>> aabf279 (All files)
def CNN3(num_kernels1 = 64, num_kernels2 = 64,num_kernels3 = 64,kernel_size=5,window_size_in=30, window_size=30, alpha=0.001, l1_coef = 0.01):
    inputs = Input(shape=X_CNN.shape[1:])
    x = Conv1D(num_kernels1, kernel_size, activation='relu', padding = 'same', kernel_regularizer=l1(l1_coef))(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(num_kernels2, kernel_size, activation='relu', padding = 'same', kernel_regularizer=l1(l1_coef))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(num_kernels3, kernel_size, activation='relu', padding = 'same', kernel_regularizer=l1(l1_coef))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    output = Dense(window_size, activation='linear')(x)

    model = Model(inputs,output)
    model.compile(optimizer = AdamW(learning_rate = alpha), loss="mse", metrics=['mae'])
    return model

# CNN + LSTM Model with L1 regularization
def CNN2_LSTM(num_hiddens=64, num_kernels1=64, num_kernels2=64, kernel_size=5, window_size_in=30, window_size=30, alpha=0.001, l1_coef=0.01):
    inputs = Input(shape=X_CNN.shape[1:])
<<<<<<< HEAD
    # First convolutional block
    x = Conv1D(num_kernels1, kernel_size, activation='relu', padding='same', kernel_regularizer=l1(l1_coef))(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    # Second convolutional block
    x = Conv1D(num_kernels2, kernel_size, activation='relu', padding='same', kernel_regularizer=l1(l1_coef))(x)
    x = MaxPooling1D(pool_size=2)(x)
    # LSTM layer
    x = LSTM(num_hiddens, return_sequences=True, kernel_regularizer=l1(l1_coef))(x)
=======
    x = Conv1D(num_kernels1, kernel_size, activation='relu', padding='same', kernel_regularizer=l1(l1_coef))(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(num_kernels2, kernel_size, activation='relu', padding='same', kernel_regularizer=l1(l1_coef))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(num_hiddens, return_sequences=False, kernel_regularizer=l1(l1_coef))(x)
>>>>>>> aabf279 (All files)
    # x = Flatten()(x)
    # Output layer - change this part to use a proper Dense layer
    output = Dense(window_size, activation='linear')(x)

    model = Model(inputs, output)
    model.compile(optimizer=AdamW(learning_rate=alpha), loss="mse", metrics=['mae'])
    return model

<<<<<<< HEAD
estimator1 = CNN3()
estimator2 = CNN2_LSTM()

estimator1.summary()
estimator2.summary()


=======
# estimator1 = NN3()
# estimator2 = LSTM3()
# estimator3 = CNN3()
# estimator4 = CNN2_LSTM()

# # estimator1.summary()
# # estimator2.summary()
# # estimator3.summary()
# # estimator4.summary()

# # # Training
# # history1 = estimator1.fit(X_CNN.reshape(17099,30*11), Y_CNN, epochs=500, batch_size=32, validation_split=0.2)
# # history2 = estimator2.fit(X_CNN, Y_CNN, epochs=500, batch_size=32, validation_split=0.2)
# # history3 = estimator3.fit(X_CNN, Y_CNN, epochs=500, batch_size=32, validation_split=0.2)
# # history4 = estimator4.fit(X_CNN, Y_CNN, epochs=500, batch_size=32, validation_split=0.2)


################################## Optuna
# import optuna
# from keras.callbacks import EarlyStopping

# early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


# def objective(trial,estimator):
#     window_size_in = 30
#     window_size = 30
#     if estimator== "CNN3":
#         num_kernels1 = trial.suggest_int('num_kernels1', 32, 128)
#         num_kernels2 = trial.suggest_int('num_kernels2', 32, 128)
#         num_kernels3 = trial.suggest_int('num_kernels3', 32, 128)
#         kernel_size  = trial.suggest_int('kernel_size', 3, 7)
#     elif estimator== "CNN2_LSTM":
#         num_kernels1 = trial.suggest_int('num_kernels1', 32, 128)
#         num_kernels2 = trial.suggest_int('num_kernels2', 32, 128)
#         kernel_size = trial.suggest_int('kernel_size', 3, 7)
#         num_hiddens = trial.suggest_int('num_hiddens', 32, 128)
#     else:
#         num_hiddens1 = trial.suggest_int('num_hiddens1', 32, 128)
#         num_hiddens2 = trial.suggest_int('num_hiddens2', 32, 128)
#         num_hiddens3 = trial.suggest_int('num_hiddens3', 32, 128)
    

#     alpha   = trial.suggest_loguniform('alpha', 1e-5, 1e-2)
#     l1_coef = trial.suggest_loguniform('l1_coef', 1e-5, 1e-2)

#     # Create the model with the suggested hyperparameters
#     if estimator == "CNN3":
#         model = CNN3(num_kernels1=num_kernels1, num_kernels2=num_kernels2, num_kernels3=num_kernels3,
#                         kernel_size=kernel_size, window_size_in=window_size_in, window_size=window_size,
#                         alpha=alpha, l1_coef=l1_coef)
#         history = model.fit(X_CNN, Y_CNN, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stop])

#     elif estimator == "CNN2_LSTM":
#         model = CNN2_LSTM(num_hiddens=num_hiddens, num_kernels1=num_kernels1, num_kernels2=num_kernels2,
#                         kernel_size=kernel_size, window_size_in=window_size_in, window_size=window_size,
#                         alpha=alpha, l1_coef=l1_coef)
#         history = model.fit(X_CNN, Y_CNN, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stop])

#     elif estimator == "LSTM3":
#         model = LSTM3(num_hiddens1=num_hiddens1, num_hiddens2=num_hiddens2, num_hiddens3=num_hiddens3,
#                         window_size_in=window_size_in, window_size=window_size,
#                         alpha=alpha, l1_coef=l1_coef)
#         history = model.fit(X_CNN, Y_CNN, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stop])

#     else:
#         # For NN3
#         model = NN3(num_hiddens1=num_hiddens1, num_hiddens2=num_hiddens2, num_hiddens3=num_hiddens3,
#                         window_size_in=window_size_in, window_size=window_size,
#                         alpha=alpha, l1_coef=l1_coef)
#         history = model.fit(X_CNN.reshape(17099,30*11), Y_CNN, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stop])

#     val_loss = history.history['val_loss'][-1]
#     # val_mae = history.history['val_mae'][-1]

#     return val_loss


# NNt = {"NN3":NN3, "LSTM3":LSTM3, "CNN3":CNN3, "CNN2_LSTM":CNN2_LSTM}

# print(f"Network is {"CNN2_LSTM"}")
# study = optuna.create_study(direction='minimize')
# study.optimize(lambda trial: objective(trial, estimator="CNN2_LSTM"), n_trials=50)
# print("Best hyperparameters: ", study.best_params)
# print("Best trial: ", study.best_trial)
# print("Best value: ", study.best_value)


# import csv

# # Example dictionary (Optuna's best_params)
# best_params = study.best_params  # This must be a dictionary

# # Write to CSV
# with open('best_params.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Parameter', 'Value'])  # Optional header
#     for key, value in best_params.items():
#         writer.writerow([key, value])
#########################################################################################################
# Load the best parameters from the CSV file
import csv

params_NN = {}
params_LSTM = {}
params_CNN = {}
params_CNN_LSTM = {}

with open('best_params_NN3.csv', 'r') as file:
    reader = csv.reader(file)
    # next(reader)  # Skip header
    for row in reader:
        key, value = row
        params_NN[key] = float(value) 

with open('best_params_LSTM.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        key, value = row
        params_LSTM[key] = float(value) 

with open('best_params_CNN3.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        key, value = row
        params_CNN[key] = float(value) 

with open('best_paramsCNN_LSTM.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        key, value = row
        params_CNN_LSTM[key] = float(value) 



estimator1 = NN3(num_hiddens1   =int(params_NN["num_hiddens1"]),
                num_hiddens2    =int(params_NN["num_hiddens2"]),
                num_hiddens3    =int(params_NN["num_hiddens3"]),
                alpha           =params_NN["alpha"],
                l1_coef         =params_NN["l1_coef"])

estimator2 = LSTM3(num_hiddens1 =int(params_LSTM["num_hiddens1"]),
                num_hiddens2    =int(params_LSTM["num_hiddens2"]),
                num_hiddens3    =int(params_LSTM["num_hiddens3"]),
                alpha           =params_LSTM["alpha"],
                l1_coef         =params_LSTM["l1_coef"])

estimator3 = CNN3(num_kernels1  = int(params_CNN["num_kernels1"]),
                num_kernels2    = int(params_CNN["num_kernels2"]),
                num_kernels3    = int(params_CNN["num_kernels3"]),
                kernel_size     = int(params_CNN["kernel_size"]),
                alpha           = params_CNN["alpha"],
                l1_coef         = params_CNN["l1_coef"])

estimator4 = CNN2_LSTM(num_hiddens      = int(params_CNN_LSTM["num_hiddens"]),
                        num_kernels1    = int(params_CNN_LSTM["num_kernels1"]),
                        num_kernels2    = int(params_CNN_LSTM["num_kernels2"]),
                        kernel_size     = int(params_CNN_LSTM["kernel_size"]),
                        alpha           = params_CNN_LSTM["alpha"],
                        l1_coef         = params_CNN_LSTM["l1_coef"])

from keras.callbacks import ModelCheckpoint


checkpoint_NN = ModelCheckpoint(
    filepath='best_model_NN.h5',        
    monitor='val_loss',              
    save_best_only=True,             
    mode='min',                      
    save_weights_only=False          
)

checkpoint_LSTM = ModelCheckpoint(
    filepath='best_model_LSTM.h5',        
    monitor='val_loss',              
    save_best_only=True,             
    mode='min',                      
    save_weights_only=False          
)

checkpoint_CNN = ModelCheckpoint(
    filepath='best_model_CNN.h5',        
    monitor='val_loss',              
    save_best_only=True,             
    mode='min',                      
    save_weights_only=False          
)

checkpoint_CNN_LSTM = ModelCheckpoint(
    filepath='best_model_CNN_LSTM.h5',        
    monitor='val_loss',              
    save_best_only=True,             
    mode='min',                      
    save_weights_only=False          
)

# # Training
history1 = estimator1.fit(X_CNN.reshape(17099,30*11), Y_CNN, epochs=1000, batch_size=32, validation_split=0.2,callbacks=[checkpoint_NN])
history2 = estimator2.fit(X_CNN, Y_CNN, epochs=1000, batch_size=32, validation_split=0.2,callbacks=[checkpoint_LSTM])
history3 = estimator3.fit(X_CNN, Y_CNN, epochs=1000, batch_size=32, validation_split=0.2,callbacks=[checkpoint_CNN])
history4 = estimator4.fit(X_CNN, Y_CNN, epochs=1000, batch_size=32, validation_split=0.2,callbacks=[checkpoint_CNN_LSTM])
import os
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mat73
import h5py
import scipy.io as scipy
print(os.getcwd())
os.chdir("/home/armin/Desktop/paper3_python/paper3_py")


data = mat73.loadmat('data.mat')
data = scipy.loadmat('data.mat')

data = data["data"]
X = np.delete(data, [6,9,11], axis=1)
Y = data[:,4]

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

MM = MinMaxScaler()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle=False)
X_train_n = MM.fit_transform(X_train)
X_test_n = MM.transform(X_test)

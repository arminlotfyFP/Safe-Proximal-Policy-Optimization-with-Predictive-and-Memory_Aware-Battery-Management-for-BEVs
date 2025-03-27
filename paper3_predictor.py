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
os.chdir("/home/armin/Desktop/paper3_python")


data = mat73.loadmat('data.mat')
data = scipy.loadmat('data.mat')
# data = h5py.File('data.mat')
# data = scipy.io.loadmat('data.mat')
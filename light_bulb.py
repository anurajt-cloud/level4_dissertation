import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import *
import os
import seaborn as sns
from sklearn import *
from sklearn.metrics import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, LeakyReLU
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
import itertools
import time
from sklearn.utils import resample
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import ECG.eager_ops
from ECG import eager_ops
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from sklearn.model_selection import train_test_split

path = './'

train_c1 = np.genfromtxt(path+'Data/normal_ex_method_0.csv', delimiter=',')
train_c0 = np.genfromtxt(path+'Data/normal_ex_method_1.csv', delimiter=',')

print(train_c0.shape)
print(train_c1.shape)
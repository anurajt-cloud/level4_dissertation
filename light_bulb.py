import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.utils import resample
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
print("starting loading")
channel_1_data = np.empty(shape=[0, 462])

c1_data = glob.glob(path+'Data/normal_ex_method_0.csv')

for j in c1_data:
    print('Loading ', j)
    csvrows = np.loadtxt(j, delimiter=',')
    channel_1_data = np.append(channel_1_data, csvrows, axis=0)

print(channel_1_data.shape)

channel_2_data = np.empty(shape=[0, 462])

c2_data = glob.glob('./Data/normal_ex_method_1.csv')

for j in c2_data:
    print('Loading ', j)
    csvrows = np.loadtxt(j, delimiter=',')
    channel_2_data = np.append(channel_2_data, csvrows, axis=0)

print(channel_2_data.shape)

def showResults(test, pred, model_name):
    accuracy = accuracy_score(test, pred)
    precision= precision_score(test, pred, average='macro')
    recall = recall_score(test, pred, average = 'macro')
    f1score_macro = f1_score(test, pred, average='macro') 
    f1score_micro = f1_score(test, pred, average='micro') 
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("f1score macro : {}".format(f1score_macro))
    print("f1score micro : {}".format(f1score_micro))
    cm=confusion_matrix(test, pred, labels=[1,2,3,4,5,6,7,8])
    return [model_name, round(accuracy,3), round(precision,3) , round(recall,3) , round(f1score_macro,3), 
            round(f1score_micro, 3), cm]

X = np.concatenate((channel_1_data[:,:-2],channel_2_data[:,:-2]), axis=1)
y = channel_1_data[:,-2]


def getCNNModel(i1, i2):
    
    input = keras.layers.Input(shape=(i1,i2))
    x = keras.layers.Conv1D(kernel_size=16, filters=32, strides=1, use_bias=True, kernel_initializer='VarianceScaling')(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    #block 2

    x = keras.layers.Conv1D(kernel_size=16, filters=32, strides=1, use_bias=True, kernel_initializer='VarianceScaling')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = keras.layers.Dropout(0.2)(x)

    #block 3
    for i in range(4):


        shortcut = MaxPooling1D(pool_size=1)(x)

        filters = 64 * ((i//2)+1)
        # print("Filter size = "+str(filters))
        x = keras.layers.Conv1D(kernel_size=16, filters=filters, strides=1, use_bias=True, padding="same", kernel_initializer='VarianceScaling')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.3)(x)

        x = keras.layers.Conv1D(kernel_size=16, filters=32, strides=1, use_bias=True, padding="same", kernel_initializer='VarianceScaling')(x)

        x = keras.layers.LeakyReLU(alpha=0.3)(x)
        x = keras.layers.Dropout(0.2)(x)
    

        x = tf.keras.layers.Add()([x, shortcut])


    x = keras.layers.Conv1D(kernel_size=16, filters=32, strides=1, use_bias=True, kernel_initializer='VarianceScaling')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = keras.layers.Flatten()(x)
    out = keras.layers.Dense(9, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[input], outputs=out)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    return model


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
k = 1
for train_ix, test_ix in kfold.split(X, y):
    train_x ,train_y = X[train_ix], y[train_ix]
    test_x ,test_y = X[test_ix], y[test_ix]
    
    N = train_x[train_y==1.0]
    L = train_x[train_y==2.0]
    R = train_x[train_y==3.0]
    V = train_x[train_y==4.0]
    A = train_x[train_y==5.0]
    F = train_x[train_y==6.0]
    f = train_x[train_y==7.0]
    I = train_x[train_y==8.0]

    seed=42
    np.random.seed(seed)
    def downsample(arr, n, seed):
        downsampled = resample(arr,replace=False,n_samples=n, random_state=seed)
        return downsampled

    def upsample(arr, n, seed):
        upsampled = resample(arr,replace=True,n_samples=n,random_state=seed)
        return upsampled

    all_class = [N,L,R,V,A,F,f,I]
    abn_class = [L,R,V,A,F,f,I]
    print("staring resampling")
    mean_val = np.mean([len(i) for i in abn_class], dtype= int)
    train_r_x = []
    train_r_y = []
    # Resampling the data
    for i in range(len(all_class)):
        if all_class[i].shape[0]> mean_val:
            all_class[i] = downsample(all_class[i],mean_val,seed)
        elif all_class[i].shape[0]< mean_val:
            all_class[i] = upsample(all_class[i], mean_val,seed)
        
        train_r_x.append(all_class[i])
        train_r_y.append(np.full(all_class[i].shape[0], i+1))
    # Shuffling and saving the data into files
    train_x_sampled = np.concatenate(train_r_x)

    train_y_sampled = np.concatenate(train_r_y)
    print(train_x_sampled.shape,train_y_sampled.shape)

    verbose = 1
    epochs = 10
    batch_size = 50

    opt = Adam(learning_rate=0.0001)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
            factor=0.1,
            patience=2,
            min_lr=0.0001 * 0.0001)

    train_x_sampled = train_x_sampled.reshape(-1, train_x_sampled.shape[1], 1).astype('float32')
    test_x = test_x.reshape(-1, test_x.shape[1], 1).astype('float32')
    print(train_x_sampled.shape[1], test_x.shape[2])
    train_y_sampled = tf.keras.utils.to_categorical(train_y_sampled)
    model = getCNNModel(train_x_sampled.shape[1],train_x_sampled.shape[2])
    h = model.fit(train_x_sampled,train_y_sampled,epochs=epochs,batch_size=batch_size,validation_split=0.1,verbose=verbose, callbacks=[reduce_lr])
    
    tl = h.history['loss']
    vl = h.history['val_loss']
    ta = h.history['accuracy']
    va = h.history['val_accuracy']

    history = np.array([tl, vl, ta, va])

    preds = np.argmax(model.predict(test_x, verbose=1),axis=1)
    
    results = showResults(test_y, preds, "Model_cv"+str(k))

    cms = confusion_matrix(test_y, preds, normalize='true')
    
    new_path = './eval_data_10k/cnn_ph/'

    model.save(new_path+'Model_cv'+str(k)+'.h5')
    np.save(new_path+"preds_cv"+str(k)+'.npy', preds)
    np.save(new_path+"results_cv"+str(k)+'.npy', results)
    np.save(new_path+"cms_cv"+str(k)+'.npy', cms)
    np.save(new_path+"history_cv"+str(k)+'.npy', history)
    tf.keras.backend.clear_session()

    
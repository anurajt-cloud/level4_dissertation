# Imports
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import *
import os
#import seaborn as sns
from sklearn import *
from sklearn.metrics import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling1D
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



# Check for GPU
print("You are using Tensorflow version", tf.__version__)

print(tf.config.list_physical_devices('GPU'))

print("-"*100)

# Defining the path
path = './'

# Loading data
train_c1 = np.genfromtxt(path+'Data/train_patients_sc.csv', delimiter=',')
train_c0 = np.genfromtxt(path+'Data/train_patients_fc.csv', delimiter=',')
test_c1 = np.genfromtxt(path+'Data/test_patients_sc.csv', delimiter=',')
test_c0 = np.genfromtxt(path+'Data/test_patients_fc.csv', delimiter=',')

train_x_c01 = np.concatenate((train_c0[:, :-2], train_c1[:, :-2]), axis=1)
train_y_c01 = np.concatenate((train_c0[:, -2:], train_c1[:, -2:]), axis=1)

test_x_c01 = np.concatenate((test_c0[:, :-2], test_c1[:, :-2]), axis=1)
test_y_c01 = np.concatenate((test_c0[:, -2:], test_c0[:, -2:]), axis=1)

train_x = train_x_c01.reshape(-1, train_x_c01.shape[1], 1).astype('float32')
test_x = test_x_c01.reshape(-1, test_x_c01.shape[1], 1).astype('float32')

train_y = tf.keras.utils.to_categorical(train_y_c01[:,0])
test_y = tf.keras.utils.to_categorical(test_y_c01[:,0])

print("Train:")
print("x:", train_x.shape, "y:", train_y.shape)
print("Test")
print("x:", test_x.shape, "y:", test_y.shape)

print("-"*100)

# Create a performance metrics function
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
    return (model_name, round(accuracy,3), round(precision,3) , round(recall,3) , round(f1score_macro,3), 
            round(f1score_micro, 3), cm)

# Function for building the LSTM model

verbose, epoch, batch_size = 1, 10, 50
activationFunction='relu'

opt = Adam(learning_rate=0.001)

def getCNNModel():
    
    input = keras.layers.Input(shape=(train_x.shape[1],train_x.shape[2]))
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

cnn1 = getCNNModel()
# cnn2 = getCNNModel()
# cnn3 = getCNNModel()
# cnn4 = getCNNModel()
# cnn5 = getCNNModel()
# cnn6 = getCNNModel()
# cnn7 = getCNNModel()
# cnn8 = getCNNModel()
# cnn9 = getCNNModel()
# cnn10 = getCNNModel()

modellist = [cnn1]#, cnn2, cnn3, cnn4, cnn5, cnn6, cnn7, cnn8, cnn9, cnn10]

# Fiiting all the models
modelhistory = []
for m in modellist:
  modelhistory.append(m.fit(train_x, train_y, epochs=epoch, verbose=verbose, validation_split=0.1, batch_size = batch_size))

print("-"*100)

# Model predictions
modelpreds = []
for m in range(len(modellist)):
  print("*"*10,"Model",m+1,"*"*10)
  modelpreds.append(modellist[m].predict(test_x, verbose=1))

print("-"*100)

# Generating the metrics

results = []
lstm_actual_value=np.argmax(test_y,axis=1)
for p in range(len(modelpreds)):
  print("*"*10,"Model",p+1,"*"*10)
  results.append(showResults(lstm_actual_value, np.argmax(modelpreds[p],axis=1),'LSTM'+str(p)))

# Generating the confusion matrices
cms = []
for p in modelpreds:
  cms.append(confusion_matrix(lstm_actual_value, np.argmax(p,axis=1), normalize='true'))

# Extracting the model train and validation accuracy & loss

modelosses = []
modeleval_losses = []
modelacc = []
modeleval_acc = []
for h in modelhistory:
  modelosses.append(h.history['loss'])
  modeleval_losses.append(h.history['val_loss'])
  modelacc.append(h.history['accuracy'])
  modeleval_acc.append(h.history['val_accuracy'])
modelosses = np.array(modelosses)
modeleval_losses = np.array(modeleval_losses)
modelacc = np.array(modelacc)
modeleval_acc = np.array(modeleval_acc)
# losses , eval_losses, accuracy, eval_accuracy
history = np.array([modelosses, modeleval_losses, modelacc, modeleval_acc])

# Saving the data

# Saving data LSTM patient leavout
new_path = path+"eval_data_10k/lstm_pl/"

modelpreds = np.array(modelpreds)
results = np.array(results)
cms = np.array(cms)
# modelhistory = np.array(modelhistory)
np.save(new_path+"lstm_modelpreds.npy", modelpreds)
np.save(new_path+"lstm_results.npy", results)
np.save(new_path+"lstm_cms.npy", cms)
np.save(new_path+"lstm_history.npy", history )

# Saving the models
sm_path = "./saved_models/cnn_pl/"
for m in range(len(modellist)):
    print("*"*10,"Model", m+1, "*"*10)
    modellist[m].save(sm_path+"Model"+str(m)+".h5")

print("DONE!")

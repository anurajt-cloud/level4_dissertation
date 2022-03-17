# Imports
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
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling1D

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

def getlstmModel():
    
    lstmmodel = Sequential()
    lstmmodel.add(LSTM(128, return_sequences=True, input_shape=(train_x.shape[1],train_x.shape[2])))
    lstmmodel.add(LSTM(9, return_sequences=True))
    lstmmodel.add(MaxPooling1D(pool_size=4,padding='same'))
    lstmmodel.add(Flatten())
    lstmmodel.add(Dense(256, activation=tf.nn.relu))    
    lstmmodel.add(Dense(128, activation=tf.nn.relu))    
    lstmmodel.add(Dense(32, activation=tf.nn.relu))
    lstmmodel.add(Dense(9, activation='softmax'))
    lstmmodel.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    # lstmmodel.summary()
    return lstmmodel
# lstmmodel = getlstmModel()

# lstmhistory= lstmmodel.fit(train_x, train_y, epochs=epoch, verbose=verbose, validation_split=0.2, batch_size = batch_size)
# lstmpredictions = lstmmodel.predict(test_x, verbose=1)

# List of models
lstm1 = getlstmModel()
lstm2 = getlstmModel()
lstm3 = getlstmModel()
lstm4 = getlstmModel()
lstm5 = getlstmModel()
lstm6 = getlstmModel()
lstm7 = getlstmModel()
lstm8 = getlstmModel()
lstm9 = getlstmModel()
lstm10 = getlstmModel()

modellist = [lstm1,lstm2,lstm3,lstm4,lstm5,lstm6,lstm7,lstm8,lstm9,lstm10]

# Fiiting all the models
modelhistory = []
for m in modellist:
  modelhistory.append(m.fit(train_x, train_y, epochs=epoch, verbose=verbose, validation_split=0.2, batch_size = batch_size))

print("-"*100)

# Model predictions
modelpreds = []
for m in range(len(modellist)):
  print("*"*10,"Model",m+1,"*"*10)
  modelpreds.append(np.argmax(modellist[m].predict(test_x, verbose=1),axis=1))

print("-"*100)

# Generating the metrics

results = []
lstm_actual_value=np.argmax(test_y,axis=1)
for p in range(len(modelpreds)):
  print("*"*10,"Model",p+1,"*"*10)
  results.append(showResults(lstm_actual_value, modelpreds[p],'LSTM'+str(p)))

# Generating the confusion matrices
cms = []
for p in modelpreds:
  cms.append(confusion_matrix(lstm_actual_value, p, normalize='true'))

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

print("DONE!")
from tabnanny import verbose
import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import numpy as np
import pandas as pd
from scipy import *
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import os
import seaborn as sns
from sklearn import *
from sklearn.metrics import *
from sklearn.utils import shuffle
import time
from ECG import eager_ops
from sklearn.svm import SVC
import tensorflow as tf

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

train_x = train_x_c01
test_x = test_x_c01

train_y = train_y_c01[:,0]
test_y = test_y_c01[:,0]

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

svc1 = SVC()

modellist = [svc1]

# Fiiting all the models
modelhistory = []
for m in enumerate(modellist):
  print("*"*10,"Model",m[0]+1,"*"*10)
  modelhistory.append(m[1].fit(train_x, train_y))

modelpreds = []
for m in range(len(modellist)):
  print("*"*10,"Model",m+1,"*"*10)
  modelpreds.append(modellist[m].predict(test_x))

results = []
for p in range(len(modelpreds)):
  print("*"*10,"Model",p+1,"*"*10)
  results.append(showResults(test_y, modelpreds[p],'SVC'+str(p)))

# Generating the confusion matrices
cms = []
for p in modelpreds:
  cms.append(confusion_matrix(test_y, p, normalize='true'))

# Saving data LSTM patient leavout
new_path = path+"eval_data_10k/svc_pl/"

modelpreds = np.array(modelpreds)
results = np.array(results)
cms = np.array(cms)
# modelhistory = np.array(modelhistory)
np.save(new_path+"modelpreds.npy", modelpreds)
np.save(new_path+"results.npy", results)
np.save(new_path+"cms.npy", cms)
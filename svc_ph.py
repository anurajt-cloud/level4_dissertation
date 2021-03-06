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

    # train_x_sampled = train_x_sampled.reshape(-1, train_x_sampled.shape[1], 1).astype('float32')
    # test_x = test_x.reshape(-1, test_x.shape[1], 1).astype('float32')
    # print(train_x_sampled.shape[1], test_x.shape[2])
    # train_y_sampled = tf.keras.utils.to_categorical(train_y_sampled)
    print("*"*15,"Model",str(k), "*"*15)
    model = SVC()
    model.fit(train_x_sampled, train_y_sampled)

    preds = model.predict(test_x)
    
    results = showResults(test_y,preds, "Model_cv"+str(k))

    cms = confusion_matrix(test_y, preds, normalize='true')
    
    new_path = './eval_data_10k/svc_ph/'

    np.save(new_path+"preds_cv"+str(k)+'.npy', preds)
    np.save(new_path+"results_cv"+str(k)+'.npy', results)
    np.save(new_path+"cms_cv"+str(k)+'.npy', cms)
    k+=1
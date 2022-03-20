import numpy as np
import os
from ECG.eager_ops import expected_gradients as eg
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

path = "./eval_data_10k/"

def cal_eg(m, dx, dy): 
  test_d = tf.data.Dataset.from_tensor_slices((dx,dy))
  test_d = test_d.batch(200)
  att = []
  for i, (x_batch, y_batch) in enumerate(test_d):
    att.append(eg(inputs=x_batch, labels=y_batch, model=m))
  return tf.stack(att)

def eg_att(foldername, tname):
    if tname=="ph":
        test_y = np.load("./eval_data_10k/teset_y.npy", allow_pickle=True)
        test_x = np.load("./eval_data_10k/teset_x.npy", allow_pickle=True)
        attributions = []
        for i in range(1,11):
            m = tf.keras.models.load_model(fp+foldername+"/Model_cv"+str(i)+".h5")
            dx = test_x[i-1].reshape(-1, test_x[i-1].shape[1], 1).astype(np.float32)
            dy = tf.keras.utils.to_categorical(test_y[i-1].astype(np.float32))
            attributions.append(cal_eg(m, dx, dy))
        return attributions
    elif tname=="pl":
        test_c0 = np.genfromtxt('./Data/test_patients_fc.csv', delimiter=',')
        test_c1 = np.genfromtxt('./Data/test_patients_sc.csv', delimiter=',')
        test_x_c01 = np.concatenate((test_c0[:, :-2], test_c1[:, :-2]), axis=1)
        test_y_c01 = np.concatenate((test_c0[:, -2:], test_c0[:, -2:]), axis=1)
        dx = test_x_c01.reshape(-1, test_x_c01.shape[1], 1).astype('float32')
        dy = tf.keras.utils.to_categorical(test_y_c01[:,0])
        m = tf.keras.models.load_model(path+foldername+"/Model0.h5")
        return cal_eg(m, dx, dy)

print(eg_att("cnn_pl", "pl").shape)
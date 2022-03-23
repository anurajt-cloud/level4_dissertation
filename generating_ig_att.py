import numpy as np
import os
from IG import *
import tensorflow as tf
import matplotlib.pyplot as plt
import time
path = "./eval_data_10k/"

def cal_ig(m, dx, dy): 
    test_d = tf.data.Dataset.from_tensor_slices((dx,dy))
    test_d = test_d.batch(200)
    att = None
    for i, (x_batch, y_batch) in enumerate(test_d):
        if i==0:
            att = calculate_ig(model=m, beats=x_batch, class_indexes=y_batch)
        else:
            att = np.append(att,calculate_ig(model=m, beats=x_batch, class_indexes=y_batch),axis=0)
        # print(att.shape)
        # att = np.append(att,eg(inputs=x_batch, labels=y_batch, model=m),axis=0)
    return att

def ig_att(foldername, tname):
    if tname=="ph":
        test_y = np.load("./eval_data_10k/teset_y.npy", allow_pickle=True)
        test_x = np.load("./eval_data_10k/teset_x.npy", allow_pickle=True)
        # attributions = []
        for i in range(5,11):
            m = tf.keras.models.load_model(path+foldername+"/Model_cv"+str(i)+".h5")
            dx = test_x[i-1].reshape(-1, test_x[i-1].shape[1], 1).astype(np.float32)
            dy = test_y[i-1].astype(np.float32)
            np.save("./ig_attributions/"+foldername+"/att"+str(i),cal_ig(m, dx, dy))
            # cal_ig(m, dx, dy)
            print(foldername+"/att"+str(i), "saved!")
    elif tname=="pl":
        test_c0 = np.genfromtxt('./Data/test_patients_fc.csv', delimiter=',')
        test_c1 = np.genfromtxt('./Data/test_patients_sc.csv', delimiter=',')
        test_x_c01 = np.concatenate((test_c0[:, :-2], test_c1[:, :-2]), axis=1)
        test_y_c01 = np.concatenate((test_c0[:, -2:], test_c0[:, -2:]), axis=1)
        dx = test_x_c01.reshape(-1, test_x_c01.shape[1], 1).astype('float32')
        dy = test_y_c01[:,0]
        m = tf.keras.models.load_model(path+foldername+"/Model0.h5")
        return cal_ig(m, dx, dy)
        # s = time.time()
        # print(cal_ig(m, dx, dy).shape, time.time()-s)


print("Generating Attributions")
# np.save("./ig_attributions/cnn_pl", ig_att("cnn_pl", "pl")) DONE!
# print("cnn_pl saved!")
# np.save("./ig_attributions/lstm_pl", ig_att("lstm_pl", "pl")) DONE!
# print("lstm_pl saved!")
# ig_att("cnn_ph", "ph") DONE!
# print("cnn_ph_eg saved!")
ig_att("lstm_ph", "ph")
# print("lstm_ph_eg saved!")
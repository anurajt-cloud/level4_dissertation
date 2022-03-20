import numpy as np
import os
from ECG.eager_ops import expected_gradients as eg
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

path = "./eval_data_10k/"

def eg_att(foldername, tname):
    test_y = np.load("./eval_data_10k/teset_y.npy", allow_pickle=True)
    test_x = np.load("./eval_data_10k/teset_x.npy", allow_pickle=True)
    # if tname=="ph":
    #     models = []
    #     attributions = []
    #     for i in range(1,11):
    #         m = tf.keras.models.load_model(path+foldername+"/Model_cv"+str(i)+".h5")
    #         att = eg(inputs=tf.convert_to_tensor(test_x[i-1].reshape(-1, test_x[i-1].shape[1], 1).astype(np.float32)), labels=tf.keras.utils.to_categorical(test_y[i-1].astype(np.float32)), model=m)
    #         print(i, test_x[i-1].shape, test_y[i-1].shape, att.shape)
    #         break
    # elif tname=="pl":
    print(test_y.shape, test_x.shape)
eg_att("cnn_ph_eg", "ph")
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import *
import os
# import seaborn as sns
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
from alibi.explainers import IntegratedGradients


print("You are using Tensorflow version", tf.__version__)

print(tf.config.list_physical_devices('GPU'))

path = './'

print("Loading Data")
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

print("Data Loaded")

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

print("Defining models")
cnn1 = getCNNModel()
cnn2 = getCNNModel()
cnn3 = getCNNModel()
cnn4 = getCNNModel()
cnn5 = getCNNModel()
cnn6 = getCNNModel()
cnn7 = getCNNModel()
cnn8 = getCNNModel()
cnn9 = getCNNModel()
cnn10 = getCNNModel()

modellist = [cnn1]#, cnn2, cnn3, cnn4, cnn5, cnn6, cnn7, cnn8, cnn9, cnn10]
print("Models defined")

lamb = 0.0001
num_epochs = 10
batch_size = 20
optimizer = tf.optimizers.Adam(learning_rate = 0.0001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
val_loss_fn = tf.keras.losses.CategoricalCrossentropy()
train_acc_fn = tf.keras.metrics.CategoricalAccuracy()
val_acc_fn = tf.keras.metrics.CategoricalAccuracy()
test_acc_fn = tf.keras.metrics.CategoricalAccuracy()

#train step
def train_step(model_m, inputs, labels):
    with tf.GradientTape() as tape:
        tape.watch(tf.convert_to_tensor(inputs))
        predictions = model_m(inputs, training=True)
        pred_loss = loss_fn(labels, predictions)
        total_loss = pred_loss

        if len(model_m.losses) > 0:
            regularization_loss = tf.math.add_n(model_m.losses)
            total_loss = total_loss + regularization_loss

        ig = IntegratedGradients(model=model_m)
        e = tf.function(ig.function)
        exp = e(X=inputs.numpy(),baselines=None,target=np.argmax(predictions.numpy(),axis=1))
        attributions = exp.attributions[0]
        
        summed_attributions = tf.reduce_sum(attributions, axis=-1, keepdims=True)

        normalized_attributions = tf.image.per_image_standardization(summed_attributions)

        attribution_loss = lamb * tf.reduce_mean(tf.image.total_variation(normalized_attributions))

        total_loss+= attribution_loss
    
    gradients = tape.gradient(total_loss, model_m.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_m.trainable_variables))
    return predictions,total_loss


# function to train a model
def training(model):
    training_acc = []
    training_loss = []
    validation_acc = []
    validation_loss = []

    start_time = time.time()
    for epoch in range(num_epochs):
        indices = np.random.permutation(len(train_x))
        # train validaion split
        train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=train_y[indices])
        # extracting the training set
        new_train_x = train_x[train_indices]
        new_train_y = train_y[train_indices]
        # generating indices for training batches
        new_indices = np.random.permutation(len(new_train_x))
        # extracting the validation set
        val_x = train_x[val_indices]
        val_y = train_y[val_indices]
        

        for i in range(0, len(new_train_x), batch_size):
            x_batch_train = new_train_x[new_indices[i:min(i + batch_size, len(new_train_x))]]
            y_batch_train = new_train_y[new_indices[i:min(i + batch_size, len(new_train_y))]]
            ts = tf.function(train_step)
            predictions,epoch_loss = ts(model, x_batch_train, y_batch_train)
            print(i)
            train_acc_fn(y_batch_train, predictions)

        train_acc = train_acc_fn.result().numpy()

        # validating
        val_preds = model.predict(val_x)
        val_acc_fn(val_y, val_preds)
        val_loss = val_loss_fn(val_y, val_preds).numpy()

        # appending training loss and acc
        training_acc.append(train_acc)
        training_loss.append(epoch_loss.numpy())
        # appending validation loss and acc
        validation_acc.append(val_acc_fn.result().numpy())
        validation_loss.append(val_loss)

        print('Epoch {} - train_accuracy: {:.4f}, train_loss: {:.4f} | val_accuracy: {:.4f}, val_loss: {:.4f} ({:.1f} seconds / epoch)'.format(epoch + 1, train_acc, epoch_loss, val_acc_fn.result().numpy(), val_loss, time.time()-start_time))

        start_time = time.time()
        train_acc_fn.reset_states()
        val_acc_fn.reset_states()
    return training_loss, validation_loss, training_acc, validation_acc

modelosses = []
modeleval_losses = []
modelacc = []
modeleval_acc = []
for m in range(len(modellist)):
    print("*"*10,"Model",m+1,"*"*10)
    tl, vl, ta, va = training(modellist[m])
    modelosses.append(tl)
    modeleval_losses.append(vl)
    modelacc.append(ta)
    modeleval_acc.append(va)

modelosses = np.array(modelosses)
modeleval_losses = np.array(modeleval_losses)
modelacc = np.array(modelacc)
modeleval_acc = np.array(modeleval_acc)
# losses , eval_losses, accuracy, eval_accuracy
history = np.array([modelosses, modeleval_losses, modelacc, modeleval_acc])

modelpreds = []
for m in range(len(modellist)):
    print("*"*10, "Model", m+1, "*"*10)
    modelpreds.append(np.argmax(modellist[m].predict(test_x, verbose=1),axis=1))

results = []
cnn_actual_value=np.argmax(test_y,axis=1)
for p in range(len(modelpreds)):
    print("*"*10,"Model",p+1,"*"*10)
    results.append(showResults(cnn_actual_value, modelpreds[p],'CNN'+str(p)))

cms = []
for p in modelpreds:
    cms.append(confusion_matrix(cnn_actual_value, p, normalize='true'))

# Saving data LSTM patient leavout
new_path = path+"eval_data_10k/cnn_pl_ig/"

modelpreds = np.array(modelpreds)
results = np.array(results)
cms = np.array(cms)
# modelhistory = np.array(modelhistory)
np.save(new_path+"cnn_modelpreds.npy", modelpreds)
np.save(new_path+"cnn_results.npy", results)
np.save(new_path+"cnn_cms.npy", cms)
np.save(new_path+"cnn_history.npy", history)

# Saving the models
sm_path = "./saved_models/cnn_pl_ig/"
for m in range(len(modellist)):
    print("*"*10,"Model", m+1, "*"*10)
    modellist[m].save(sm_path+"Model"+str(m))

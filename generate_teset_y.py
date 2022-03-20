import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold

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


X = np.concatenate((channel_1_data[:,:-2],channel_2_data[:,:-2]), axis=1)
y = channel_1_data[:,-2]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
k = 1
c1 = []
c1_x = []
for train_ix, test_ix in kfold.split(X, y):
    train_x ,train_y = X[train_ix], y[train_ix]
    test_x ,test_y = X[test_ix], y[test_ix]
    print(test_y.shape)
    c1.append(test_y)
    c1_x.append(test_x)

np.save("./eval_data_10k/teset_x", c1_x,allow_pickle=True)
# np.save("./eval_data_10k/teset_y", c1, allow_pickle=True)
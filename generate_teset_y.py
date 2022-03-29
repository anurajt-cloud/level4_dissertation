import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
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
c1_y = []
resamp_train_y = []
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
    c1.append(test_y)
    c1_y.append(train_y)
    resamp_train_y.append(train_y_sampled)

# Generating the CV datasets
# np.save("./eval_data_10k/teset_x", c1_x,allow_pickle=True)
# np.save("./eval_data_10k/trset_y", c1_y,allow_pickle=True)
# np.save("./eval_data_10k/teset_y", c1, allow_pickle=True)
np.save("./eval_data_10k/resamp_trset_y", resamp_train_y, allow_pickle=True)
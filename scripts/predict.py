import numpy as np
import tensorflow as tf
import pandas as pd

X_test = np.load("/home/scycw2/training_data/resNet34+-50/raw_data_50/X_train.npy")
Y_test = np.load("/home/scycw2/training_data/resNet34+-50/raw_data_50/Y_train.npy")

model = tf.keras.models.load_model('/home/scych2/training_data/resNet34+-50-ba32/saved_model/')
cp ="/home/scych2/training_data/resNet34+-50in-ba128/checkpoints/epoch_10-cp.ckpt"
model.load_weights(cp)

f = model.predict(X_test)
f = f.reshape(-1)
print("f is:" )
print(f.shape)
print(f)
# y  =  np.repeat(Y_test, 50)
y = Y_test
print("y is:" )
print(y.shape)
print(y)


#all testing R2
y_bar = np.mean(y)
print("y_bar is: " + str(y_bar))
SS_res_all = np.sum(np.square(y-f))
SS_tot_all = np.sum(np.square(y-y_bar))
R2 = 1-(SS_res_all/SS_tot_all)

print("R^2:" + str(R2))

## Pearson r
def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis

def ss(a, axis=0):
    a, axis = _chk_asarray(a,axis)
    return np.sum(a*a, axis)

#all testing R2
f_bar = np.mean(f)
y_bar = np.mean(y)

fm, ym = f-f_bar, y-y_bar
r_num = np.add.reduce(fm*ym)
r_den = np.sqrt(ss(fm) * ss(ym))
r = r_num/r_den

print("Pearson r:"+str(r))

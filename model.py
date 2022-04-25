import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import os

from tensorflow.keras import datasets, layers, models


batch_size = 4
module_count = 1
frame_count = 10
depth = 109
height = 104
width = 85
input_channels = 2
input_shape =(module_count,depth,height,width,input_channels)

instanceCount = 5
split = 0.8

model_name = 'baseline-' + str(instanceCount)
save_path = 'training_data/' + model_name + '/'

def read_single_frame(path,depth, height, width):
    with open(path, "rb") as f:
        npData = np.load(f)
    frameData = npData.reshape((depth,height,width),order='F') # Using order F to maintain x-y-z order
    return frameData

def read_trajactory(path, field, instanceCount):
    frames = []
    files = os.listdir(path)
    files.sort()
    for file_ in files:
        f_name = str(file_)
        if len(frames) >= instanceCount:
            frames = np.stack((frames), axis=0)
            return frames
        elif field == 1 and 'fld-01' in f_name:
            frames.append(read_single_frame(path+f_name, depth, height, width))
        elif field == 2 and 'fld-02' in f_name:
            frames.append(read_single_frame(path+f_name, depth, height, width))
    frames = np.stack((frames), axis=0)
    return frames

def read_y_data():
    data = pd.read_csv("data/bf.csv", sep=",")
    n = data.values
    return n

def load_x_data(path, split, instanceCount):
    trainFrameCount = int(instanceCount*split)

    stericMIF = read_trajactory(path, 1,instanceCount)
    elecMIF = read_trajactory(path, 2,instanceCount)
    X = np.stack((stericMIF, elecMIF), axis=4)

    # x_train = X[:trainFrameCount]
    # x_test = X[trainFrameCount:]
    
    # return x_train, x_test
    return X

def load_y_data(target, split, instanceCount):
    trainFrameCount = int(instanceCount*split)
    
    Y = np.ones((instanceCount,1))*target

    # y_train = Y[:trainFrameCount]
    # y_test = Y[trainFrameCount:]

    # return y_train,y_test
    return Y

def baseline_model():
    model = models.Sequential()
    model.add(layers.Conv3D(filters=2, kernel_size=3, activation='relu', input_shape=(depth,height,width,input_channels)))
    model.add(layers.MaxPooling3D(pool_size=2))
    model.add(layers.Conv3D(filters=4, kernel_size=2, activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='sigmoid'))
    model.add(layers.Dense(1))

    model.compile(optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['mse','mae',tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))])
    return model

def iter(model, x_train,y_train, x_test, y_test):

    checkpoint_path = save_path+"checkpoints/epoch_{epoch:02d}-loss_{val_loss:.2f}_cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    csv_output_path = save_path+"log.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(csv_output_path, separator=',', append=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", min_delta=1e-3,patience = 3, verbose = 1)


    # Create a callback that saves the models' weight
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    history = model.fit(x_train, 
                        y_train, 
                        epochs=3, 
                        validation_data=(x_test, y_test),                        
                        callbacks=[cp_callback,earlystop,csv_logger])
    
    return model

def prepareData(data, split,instanceCount):

    tf.random.set_seed(12)
    data_shuffled = tf.random.shuffle(data)
    
    trainFrameCount = int(instanceCount*split)
    data_train = data_shuffled[:trainFrameCount]
    data_test = data_shuffled[trainFrameCount:]

    return data_train, data_test

def train():
    model = baseline_model()
    labels = read_y_data()
    # X_train = []
    # Y_train = []
    # X_test = []
    # Y_test = []
    X = []
    Y = []
    for module,target in labels:
        path = 'data/b' + module + '/'
        print("\n=============================================")
        print("Training on "+path+" ...")
        print("=============================================\n")
        if not os.path.exists(path):
            continue
        # x_train, x_test = load_x_data(path,split,instanceCount)
        X.append(load_x_data(path,split,instanceCount))
        # y_train, y_test = load_y_data(target,split,instanceCount)
        Y.append(load_y_data(target,split,instanceCount))

        # X_train.append(x_train)
        # Y_train.append(y_train)

        # X_test.append(x_test)
        # Y_test.append(y_test)

    X_train, X_test = prepareData(np.vstack(X), split, instanceCount)

    Y = np.vstack(Y)
    Y = (Y - min(Y))/(max(Y)-min(Y))
    Y_train, Y_test = prepareData(Y, split, instanceCount)

    # Xtr = np.vstack((X_train))
    # Ytr = np.vstack((Y_train))
    # Ytr = (Ytr - min(Ytr))/(max(Ytr)-min(Ytr))
    # tf.random.set_seed(12)
    # Xtr = tf.random.shuffle(Xtr)
    # tf.random.set_seed(12)
    # Ytr = tf.random.shuffle(Ytr)

    # Xte = np.vstack((X_test))
    # Yte = np.vstack((Y_test))
    # Yte = (Yte - min(Yte))/(max(Yte)-min(Yte))
    # tf.random.set_seed(12)
    # Xte = tf.random.shuffle(Xte)
    # tf.random.set_seed(12)
    # Yte = tf.random.shuffle(Yte)


    model = iter(model, X_train, Y_train, X_test, Y_test)
    model_path = save_path+'saved_model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(model_path)


train()


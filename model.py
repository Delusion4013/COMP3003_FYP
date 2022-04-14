import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle

from tensorflow.keras import datasets, layers, models
# import matplotlib.pyplot as plt


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

def prepare_x_data(path, split, instanceCount):
    trainFrameCount = int(instanceCount*split)

    stericMIF = read_trajactory(path, 1,instanceCount)
    elecMIF = read_trajactory(path, 2,instanceCount)
    X = np.stack((stericMIF, elecMIF), axis=4)

    x_train = X[:trainFrameCount]
    x_test = X[trainFrameCount:]
    
    return x_train, x_test

def prepare_y_data(target, split, instanceCount):
    trainFrameCount = int(instanceCount*split)
    
    Y = np.ones((instanceCount,1))*target
    y_train = Y[:trainFrameCount]
    y_test = Y[trainFrameCount:]

    return y_train,y_test


def baseline_model():
    model = models.Sequential()
    model.add(layers.Conv3D(filters=2, kernel_size=3, activation='relu', input_shape=(depth,height,width,input_channels)))
    model.add(layers.MaxPooling3D(pool_size=2))
    model.add(layers.Conv3D(filters=4, kernel_size=2, activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['mse','mae'])
    return model

def iter(model, module, x_train,y_train, x_test, y_test):

    checkpoint_path = "baseline_checkpoints/"+ module + "/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    # Create a callback that saves the models' weight
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    history = model.fit(x_train, 
                        y_train, 
                        epochs=10, 
                        validation_data=(x_test, y_test),                        
                        callbacks=[cp_callback])

    # Save history of training
    history_path = 'saved_history/'+module+'_baseline.txt'
    history_dir = os.path.dirname(history_path)

    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    with open(history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    return model

def train():
    model = baseline_model()
    labels = read_y_data()
    for module,target in labels:
        path = 'data/b' + module + '/'
        print("\n=============================================")
        print("Training on "+path+" ...")
        print("=============================================\n")
        if not os.path.exists(path):
            continue
        x_train, x_test = prepare_x_data(path,split,instanceCount)
        y_train, y_test = prepare_y_data(target/10,split,instanceCount)
        model = iter(model, module, x_train, y_train, x_test, y_test)
    model_path = 'saved_model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save('saved_model/baseline')


train()


import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import os
# import pickle

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

instanceCount = 20
split = 0.8

model_name = 'pafuncy-' + str(instanceCount)
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


def pafuncy_model():
    input = tf.keras.layers.Input((109, 104, 85, 2))

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(input)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(100, activation = 'relu')(x)
    x = tf.keras.layers.Dense(50, activation = 'relu')(x)
    x = tf.keras.layers.Dense(20, activation = 'relu')(x)

    output = tf.keras.layers.Dense(units=1, activation="relu")(x)

    model = tf.keras.Model(input, output, name="pafnucylike_CNN")

    return model

def iter(model, x_train,y_train, x_test, y_test):

    checkpoint_path = save_path+"checkpoints/-epoch_{epoch:02d}-loss_{val_loss:.2f}_cp.ckpt"
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
                        epochs=100, 
                        validation_data=(x_test, y_test),                        
                        callbacks=[cp_callback,earlystop,csv_logger])

    
    return model

def train():
    model = pafuncy_model()
    model.compile(
        optimizer='adam',
        loss = 'mse',
        metrics = ['mae', tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))]
    )  
    print("Started Training...\n\n")
    X_train = np.load("/home/scycw2/training_data/resNet34-20/raw_data_20/X_train.npy")
    X_test = np.load("/home/scycw2/training_data/resNet34-20/raw_data_20/X_test.npy")
    Y_train = np.load("/home/scycw2/training_data/resNet34-20/raw_data_20/Y_train.npy")
    Y_test = np.load("/home/scycw2/training_data/resNet34-20/raw_data_20/Y_test.npy")


    model = iter(model, X_train, Y_train, X_test, Y_test)
    model_path = save_path+'saved_model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(model_path)

train()


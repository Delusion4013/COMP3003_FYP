from tkinter import X
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import os
# import pickle

from tensorflow.keras import datasets, layers, models
# import matplotlib.pyplot as plt


depth = 109
height = 104
width = 85
input_channels = 2

instanceCount = 50
split = 0.8

model_name = 'Inception5-' + str(instanceCount)
save_path = 'training_data/' + model_name + '/'

def Naive_InceptionBlk(x, f1, f2, f3):
    conv1 = tf.keras.layers.Conv3D(filters=f1, kernel_size=1, padding='same', activation="relu")(x)
    conv3 = tf.keras.layers.Conv3D(filters=f2, kernel_size=3, padding='same', activation="relu")(x)
    conv5 = tf.keras.layers.Conv3D(filters=f3, kernel_size=5, padding='same', activation="relu")(x)
    pool = tf.keras.layers.MaxPool3D(pool_size=3, strides=1,padding='same')(x)
    layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)

    return layer_out

def InceptionBlk(x, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	conv1 = tf.keras.layers.Conv3D(f1, kernel_size=1, padding='same', activation='relu')(x)
	# 3x3 conv
	conv3 = tf.keras.layers.Conv3D(f2_in, kernel_size=1, padding='same', activation='relu')(x)
	conv3 = tf.keras.layers.Conv3D(f2_out, kernel_size=3, padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = tf.keras.layers.Conv3D(f3_in, kernel_size=1, padding='same', activation='relu')(x)
	conv5 = tf.keras.layers.Conv3D(f3_out, kernel_size=5, padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = tf.keras.layers.MaxPool3D(pool_size=3, strides=1, padding='same')(x)
	pool = tf.keras.layers.Conv3D(f4_out, kernel_size=1, padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

def Inception3():
    input = tf.keras.layers.Input((109, 104, 85, 2))
    
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=7, strides=2, padding='same', activation="relu")(input)
    x = tf.keras.layers.MaxPool3D(pool_size=3, padding='same')(x)

    x = tf.keras.layers.Conv3D(filters=192, kernel_size=3, strides=1, padding='same', activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2, padding='same')(x)

    x = InceptionBlk(x, 64,96,128,16,32,32)
    x = InceptionBlk(x, 128,128,192,32,96,64)

    x = tf.keras.layers.MaxPool3D(pool_size=7, strides=1, padding='same')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(input, output, name="Inception3")

    return model

def Inception4():
    input = tf.keras.layers.Input((109, 104, 85, 2))
    
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=7, strides=2, padding='same', activation="relu")(input)
    x = tf.keras.layers.MaxPool3D(pool_size=3, padding='same')(x)

    x = tf.keras.layers.Conv3D(filters=192, kernel_size=3, strides=1, padding='same', activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2, padding='same')(x)

    x = InceptionBlk(x, 64,96,128,16,32,32)
    x = InceptionBlk(x, 128,128,192,32,96,64)
    x = tf.keras.layers.MaxPool3D(pool_size=2, padding='same')(x)

    x = InceptionBlk(x, 192,96,208,16,48,64)
    x = InceptionBlk(x, 160,112,224,24,64,64)
    x = InceptionBlk(x, 128,128,256,24,64,64)
    x = InceptionBlk(x, 112,144,288,32,64,64)
    x = InceptionBlk(x, 256,160,320,32,128,128)

    x = tf.keras.layers.MaxPool3D(pool_size=7, strides=1, padding='same')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(input, output, name="Inception4")

    return model

def Inception5():
    input = tf.keras.layers.Input((109, 104, 85, 2))
    
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=7, strides=2, padding='same', activation="relu")(input)
    x = tf.keras.layers.MaxPool3D(pool_size=3, padding='same')(x)

    x = tf.keras.layers.Conv3D(filters=192, kernel_size=3, strides=1, padding='same', activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2, padding='same')(x)

    x = InceptionBlk(x, 64,96,128,16,32,32)
    x = InceptionBlk(x, 128,128,192,32,96,64)
    x = tf.keras.layers.MaxPool3D(pool_size=2, padding='same')(x)

    x = InceptionBlk(x, 192,96,208,16,48,64)
    x = InceptionBlk(x, 160,112,224,24,64,64)
    x = InceptionBlk(x, 128,128,256,24,64,64)
    x = InceptionBlk(x, 112,144,288,32,64,64)
    x = InceptionBlk(x, 256,160,320,32,128,128)
    x = tf.keras.layers.MaxPool3D(pool_size=2, padding='same')(x)

    x = InceptionBlk(x, 256,160,320,32,128,128)
    x = InceptionBlk(x, 384,192,384,48,128,128)

    x = tf.keras.layers.MaxPool3D(pool_size=7, strides=1, padding='same')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1000, activation="sigmoid")(x)
    output = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(input, output, name="Inception5")

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
    model = Inception5()
    model.compile(
        optimizer='adam',
        loss = 'mse',
        metrics = ['mae', tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))]
    )  
   
    X_train = np.load("/home/scycw2/training_data/resNet34+-50/raw_data_50/X_train.npy")
    X_test = np.load("/home/scycw2/training_data/resNet34+-50/raw_data_50/X_test.npy")
    Y_train = np.load("/home/scycw2/training_data/resNet34+-50/raw_data_50/Y_train.npy")
    Y_test = np.load("/home/scycw2/training_data/resNet34+-50/raw_data_50/Y_test.npy")


    model = iter(model, X_train, Y_train, X_test, Y_test)
    model_path = save_path+'saved_model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(model_path)


train()


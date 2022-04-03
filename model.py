import tensorflow as tf
import numpy as np
import os

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


batch_size = 4
module_count = 1
frame_count = 10
depth = 109
height = 104
width = 85
input_channels = 2
input_shape =(module_count,depth,height,width,input_channels)

# x_train = np.random.rand(100,depth,height,width,input_channels)
# y_train = np.random.rand(100,1)

x_test = np.random.rand(20,depth,height,width,input_channels)
y_test = np.random.rand(20,1)


def read_single_frame(path,depth, height, width):
    with open(path, "rb") as f:
        npData = np.load(f)
    frameData = npData.reshape((depth,height,width),order='F') # Using order F to maintain x-y-z order
    return frameData

def read_trajactory(path, field):
    frames = []
    files = os.listdir(path) #采用listdir来读取所有文件
    files.sort() #排序
    for file_ in files:     #循环读取每个文件名
        f_name = str(file_)
        if field == 1 and 'fld-01' in f_name:
            frames.append(read_single_frame(path+f_name, depth, height, width))
        elif field == 2 and 'fld-02' in f_name:
            frames.append(read_single_frame(path+f_name, depth, height, width))
        # else:
        #     print('Invalid field number')
    frames = np.stack((frames), axis=0)
    return frames

def baseline_model():
    model = models.Sequential()
    model.add(layers.Conv3D(filters=2, kernel_size=3, activation='relu', input_shape=(depth,height,width,input_channels)))
    model.add(layers.MaxPooling3D(pool_size=2))
    model.add(layers.Conv3D(filters=4, kernel_size=2, activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='relu'))
    model.add(layers.Dense(1))

    return model

def train():
    model = baseline_model()
    model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, 
                        validation_data=(x_test, y_test))

path = 'data/b2yel/'
stericMIF = read_trajactory(path, 1)
elecMIF = read_trajactory(path, 2)
x_train = np.stack((stericMIF, elecMIF), axis=4)
y_train = np.ones((10,1))*0.7809668
train()